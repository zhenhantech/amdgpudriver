
/sys/class/drm/card*/device/current_compute_partition          # SPX
/sys/class/drm/card*/device/available_compute_partition        # SPX, DPX, CPX
/sys/class/drm/card*/device/current_memory_partition           # NPS1
/sys/class/drm/card*/device/available_memory_partition         # NPS1, NPS4



Use "umr -cpc" command to dump MEC status.

Get the firmware's instruction pointer. In below example, it is 0x47a. Correlate this instruction pointer with the MEC firmware FP32 dump to get what MEC firmware is doing when the issue happens.

 
ME 1 Pipe 0: INSTR_PTR 0x47a  INT_STAT_DEBUG 0x4000000
Pipe 0  Queue 2  VMID 8
  PQ BASE 0x7f6c61920000  RPTR 0x10  WPTR 0x10  RPTR_ADDR 0x7f6c61a04080  CNTL 0xc01c50d
  EOP BASE 0x7f6c619be000  RPTR 0x40000000  WPTR 0x3f70000  WPTR_MEM 0x8
  MQD 0xa02800  DEQ_REQ 0x0  IQ_TIMER 0x0  AQL_CONTROL 0x1
  SAVE BASE 0x0  SIZE 0x0  STACK OFFSET 0x0  SIZE 0x0
ME 1 Pipe 1: INSTR_PTR 0x47a  INT_STAT_DEBUG 0x4000000
Pipe 1  Queue 2  VMID 8
  PQ BASE 0x7f6c619c0000  RPTR 0x20  WPTR 0x20  RPTR_ADDR 0x7f6c61a32080  CNTL 0xc01c50f
  EOP BASE 0x7f6c61a23000  RPTR 0x40000010  WPTR 0x3ff8010  WPTR_MEM 0x10
  MQD 0xa01e00  DEQ_REQ 0x0  IQ_TIMER 0x0  AQL_CONTROL 0x1
  SAVE BASE 0x0  SIZE 0x0  STACK OFFSET 0x0  SIZE 0x0
ME 1 Pipe 2: INSTR_PTR 0x47a  INT_STAT_DEBUG 0x4000000
ME 1 Pipe 3: INSTR_PTR 0x47a  INT_STAT_DEBUG 0x4000000
ME 2 Pipe 0: INSTR_PTR 0x47a  INT_STAT_DEBUG 0x0
Pipe 0  Queue 0  VMID 0
  PQ BASE 0xa00000  RPTR 0x94  WPTR 0x94  RPTR_ADDR 0xa01800  CNTL 0xc0008508
  EOP BASE 0xa00800  RPTR 0x40000000  WPTR 0x3ff8000  WPTR_MEM 0x0
  MQD 0x10847dd1000  DEQ_REQ 0x0  IQ_TIMER 0x0  AQL_CONTROL 0x0
  SAVE BASE 0x0  SIZE 0x0  STACK OFFSET 0x0  SIZE 0x0
ME 2 Pipe 1: INSTR_PTR 0x47a  INT_STAT_DEBUG 0x0
Pipe 1  Queue 0  VMID 0
  PQ BASE 0x455000  RPTR 0x6b00  WPTR 0x6b00  RPTR_ADDR 0x4005c0  CNTL 0xc0308011
  EOP BASE 0x454000  RPTR 0x40000000  WPTR 0x3ff8000  WPTR_MEM 0x0
  MQD 0xf400a0a000  DEQ_REQ 0x0  IQ_TIMER 0x0  AQL_CONTROL 0x0
  SAVE BASE 0x0  SIZE 0x0  STACK OFFSET 0x0  SIZE 0x0


Other Links
I'm gradually restructuring documentation into shorter articles, linked below.

Bird's-Eye View of an AQL Dispatch

Table of Contents
Other Links
Table of Contents
Preface
Documentation
Debugging Tools
UMR (User-Mode Register Debugger)
ASIC Discovery
GFXOFF (Raven/gfx10+)
Register Access
Reading
Index Registers
Writing
Missing Registers
SR-IOV
Memory Access
Wavefront Access
CPC Dump
SDMA Dump
KFD Debugfs
KFD Dynamic Debug
HDT (Hardware Debug Tool)
Debugging Triage and Techniques
Memory
Physical Memory
Virtual Memory
Page Tables
VMID 0
Apertures
VMIDs 1-15
Large Pages
Address Translation Faults
Interrupt and Logging
Fault vs Aperture Violation
Caching
Data Caching
Address Translation Caching
CP
Overview
Hardware Queue Descriptor (HQD)
Memory Queue Descriptor (MQD)
Privileged CP Interface (HIQ, KIQ)
Process/Queue Scheduler (HWS)
User-Mode Compute Queues
Dispatch Lifecycle
User-Mode SDMA Queues
Preface
This document is intended to assist the ROCm driver developer in understanding and debugging interactions between the software and hardware. Topics covered include:

Kernel- and user-mode driver/hardware interactions in the ROCm compute stack
Key hardware processes in the compute dispatch pipeline
Debugging tools and triage methodologies for failures, including hangs and VM faults
This document does not cover:

Software driver design and architecture
Higher-level language runtimes and compilers
Documentation
Hardware specifications are, unfortunately, very sparse in details and scattered within a large repository. They are often written as "delta" documents, describing only changes since the previous generation. This is not very useful when starting from a point of no understanding.

There are three key document repositories for the driver developer:

The register specification (described below): http://adcweb02.amd.com/orlvalid/regspec/
Documents driver-accessible registers for every ASIC
Perforce repository: http://p4web.amd.com:1677
Most specifications can be found within this jungle of directories
Start e.g. at //gfxip/gfx9/doc/design for gfx blocks or //sysip/oss/oss-4.0/sdma for SDMA
Can also be browsed with a Perforce client on svvp4p01:1677
Note that these are sensitive documents which must not leave AMD systems
UTC/VM specifications: https://amdcloud.sharepoint.com/sites/gmc/Shared%20Documents/Forms/AllItems.aspx
Debugging Tools
UMR (User-Mode Register Debugger)
Internal source: http://git.amd.com:8080/gitweb?p=brahma%2Fec%2Fumr.git;a=summary
Public source: https://gitlab.freedesktop.org/tomstdenis/umr (public ASICs only)
Branches: master (public ASICs), npi (non-public ASICs)

UMR can be compiled from source with CMake. It has an optional dependency on LLVM for shader disassembly. Linking UMR statically allows it to be used on installations without the requisite libraries.

cd umr
mkdir build
cd build
cmake -DCMAKE_C_FLAGS=-static -DCMAKE_CXX_FLAGS=-static -DLLVM_DIR=/path/to/llvm/prefix ..
make
A maintained build can be downloaded from:  umr-npi.tar.gz (NPI only, not for public release)

Caution: Due to a bug in libpciaccess the --pci option will only function correctly in the above linked build.

ASIC Discovery
By default UMR interfaces with the GPU via amdgpu debugfs node 0 (/sys/kernel/debug/dri/0/amdgpu_*). On many systems DRI node 0 is an integrated GPU. A different node can be selected with the -i option:

rocm-smi
[..]
GPU  Temp   AvgPwr  SCLK     MCLK    Fan     Perf  PwrCap  VRAM%  GPU%
1    24.0c  10.0W   1269Mhz  945Mhz  97.65%  auto  220.0W    0%   0%
 
umr     
ERROR: Device 0x2000 not found in UMR device table
ASIC not found (instance=0, did=ffffffffffffffff)
 
umr -i 1
If the correct DRI node has an unrecognized PCI DID (common in SR-IOV and emulation) the ASIC type can be selected with the -f option:

umr -i 1 -f vega10
You can also select the GPU by PCI bus address:

umr --pci 00:84:00.0
In some cases it may be necessary to bypass the DRI node. For example, the PF of a SR-IOV configuration has no corresponding node. Here, the GPU must be selected by PCI address, the ASIC type specified manually, and the -O no_kernel option provided.

umr --pci 00:84:00.0 -f vega10 -O no_kernel
Caution: This option should not be used when the driver is loaded since register access is not synchronized.

Caution: The no_kernel option does not interact well with wavefront access (-wa).

GFXOFF (Raven/gfx10+)
GFXOFF is a power management feature which powers down hardware blocks when the engine is idle. In this state many UMR functions will fail. In particular all register reads (by the user or internally) will return 0xFFFFFFFF and writes will be discarded.

Before using UMR with an ASIC with GFXOFF support this feature should be disabled:

umr -go 0
Register Access
Per-ASIC register specifications: http://adcweb02.amd.com/orlvalid/regspec/

Caution: Some documentation on this site is out of date. When in doubt take the URL, remove the last component, and check the modification date. e.g. See the directory listing at: http://adcweb02.amd.com/orlvalid/regspec/web_regspec/navi10/

Registers are a fundamental interface to the GPU hardware. They are exposed through a PCI BAR on the physical function or virtual function (for SR-IOV):

lspci -v | less
[..]
02:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Ellesmere [Radeon RX 470/480/570/570X/580/580X] (rev c7) (prog-if 00 [VGA controller])
        Subsystem: Advanced Micro Devices, Inc. [AMD/ATI] Ellesmere [Radeon RX 470/480/570/580]
        Physical Slot: 17
        Flags: fast devsel, IRQ 11, NUMA node 0
        Memory at bfe0000000 (64-bit, prefetchable) [disabled] [size=256M]
        Memory at bff0000000 (64-bit, prefetchable) [disabled] [size=2M]
        I/O ports at 6000 [disabled] [size=256]
        Memory at d0200000 (32-bit, non-prefetchable) [disabled] [size=256K]
(The other two memory BARs are for VRAM and doorbell access.)

The contents of this BAR are documented for each ASIC at the link above.

Below is the entry for a 32-bit register exposed by the GRBM block at byte offset 0x8010 inside the register BAR. There are 16 instances of this register. 1 in the physical function BAR (GpuF0Reg) and 15 in the virtual function BARs (GpuF0VFxxReg).

GRBM:GRBM_STATUS  ·  [R]  ·  32 bits  ·  Access: 32  ·  GpuF0Reg:0x8010, GpuF0VF0Reg:0x8010, GpuF0VF10Reg:0x8010, GpuF0VF11Reg:0x8010, GpuF0VF12Reg:0x8010, GpuF0VF13Reg:0x8010, GpuF0VF14Reg:0x8010, GpuF0VF15Reg:0x8010, GpuF0VF1Reg:0x8010, GpuF0VF2Reg:0x8010, GpuF0VF3Reg:0x8010, GpuF0VF4Reg:0x8010, GpuF0VF5Reg:0x8010, GpuF0VF6Reg:0x8010, GpuF0VF7Reg:0x8010, GpuF0VF8Reg:0x8010, GpuF0VF9Reg:0x8010
DESCRIPTION: Status Register For GRBM and consolidated Graphics Targets
Field Name	Bits	Default	Description
ME0PIPE0_CMDFIFO_AVAIL	3:0	none	The number of available entries in the Command Processor Graphics (CPG) Microengine 0 Pipe 0 Command FIFO.


RSMU_RQ_PENDING	5	none	There is a RSMU request pending in the GRBM.


ME0PIPE0_CF_RQ_PENDING	7	none	There is a Command Processor Graphics (CPG) Microengine 0 Pipe 0 Command Fifo request pending in the GRBM.


ME0PIPE0_PF_RQ_PENDING	8	none	There is a Command Processor Graphics (CPG) Microengine 0 Pipe 0 Pre-Fetch Parser request pending in the GRBM.
It has several fields which indicate block busy status across the ASIC. This register can be read but not written (note [R] in the title). Other registers can be read and/or written.

Caution: Some registers are indexed by other registers and will read/write different values depending on the value written to the index register. e.g. Most CP registers are indexed by the GRBM_GFX_CNTL register and most SQ registers are indexed by the GRBM_GFX_INDEX register. These relationships are not documented in the register spec.

In order to access this register a process must have the register BAR mapped into its virtual address space. The KFD maintains a register BAR mapping in the kernel address space and exposes it through the privileged debugfs interface at /sys/kernel/debug/dri/N/amdgpu_regs, where N is the GPU number (shown by rocm-smi). This file can be seeked/read/written with the byte register offset.

The amdgpu_regs file descriptor also provides access to some indexed address spaces (e.g. SQIND) but this interface is primarily used by UMR.

Reading
UMR simplifies access to the register BAR. For example, we can read the GRBM_STATUS register like this:

umr -O bits -r vega10.gfx90.mmGRBM_STATUS
0x00003028
        .ME0PIPE0_CMDFIFO_AVAIL[0:3]                                     ==        8 (0x00000008)
        .RSMU_RQ_PENDING[5:5]                                            ==        1 (0x00000001)
        .ME0PIPE0_CF_RQ_PENDING[7:7]                                     ==        0 (0x00000000)
[..]
Notice how the full 32-bit value is shown (0x3028) and the fields are also shown individually. Breaking down the command-line parameters:

-O bits
Comma-separated list of options. The ‘bits’ option in this context shows the individual register fields in addition to the 32-bit value.

-r vega10.gfx90.mmGRBM_STATUS
Read the register with the given path.

UMR organizes registers into ASIC/block/register paths. There are two command-line options to discover blocks and register names:

umr -lb
        vega10.gfx90
        vega10.uvd70
        vega10.vce40
[..]
umr -lr vega10.gfx90
        vega10.gfx90.mmGRBM_CNTL => 0x02000
        vega10.gfx90.mmGRBM_SKEW_CNTL => 0x02001
        vega10.gfx90.mmGRBM_STATUS2 => 0x02002
[..]
Piping the output of -lr into grep can help locate the full path for a given register.

Index Registers
UMR provides two command-line options to adjust the GRBM_GFX_CNTL and GRBM_GFX_INDEX index registers. It is not safe to write to these registers directly because the kernel may change them after you do.

umr -h
[..]
*** Bank Selection ***
 
        --bank, -b <se> <sh> <instance>
                Select a GRBM se/sh/instance bank in decimal. Can use 'x' to denote broadcast.
 
        --sbank, -sb <me> <pipe> <queue> [vmid]
                Select a SRBM me/pipe/queue bank in decimal.  VMID is optional (default: 0).
These options are used in conjunction with the read/write options to select a specific index for a target register.

Writing
Writing to a register is less common than reading. It is generally used to adjust the hardware for specific debug configurations, or to adjust an index register not handled by UMR.

You can write a 32-bit value to any register with the -w option:

umr -r vega10.gfx90.mmSQ_IND_INDEX
0x00000000
umr -w vega10.gfx90.mmSQ_IND_INDEX 10
umr -r vega10.gfx90.mmSQ_IND_INDEX
0x00000010
Note that the value is always specified in hexadecimal.

You can also write to an individual field in the register with the -wb option. UMR will read/modify/write the register with the given value, shifted into the field’s position.

umr -wb vega10.gfx90.mmSQ_IND_INDEX.SIMD_ID 2
vega10.gfx90.mmSQ_IND_INDEX.SIMD_ID <= 0x00000020
Missing Registers
UMR’s register database is built from publicly available headers. These do not include all registers. If a register you need is unavailable it can still be accessed directly by address. However, individual fields cannot be shown. e.g. To read the GRBM_STATUS register shown earlier we can use the 0x8010 offset:

umr -r 0x8010
0x00003028
Caution: This method is not compatible with the -b or -sb options.

SR-IOV
Direct register access in virtual functions (VFs) is not allowed. Instead the driver routes all register requests from the DRI node to the CP via the KIQ (kernel interface queue). This ensures register access is synchronized with world switches.

As a consequence register access from a VF is quite slow. It may be preferable to work on the PF when accessing large amounts of registers (e.g. in wavefront access, see below).

Memory Access
All GPU-visible memory (system and VRAM) is accessible through ptrace in the owning process. Sometimes it may be useful to access this memory out-of-process, e.g. when ptrace cannot attach because an ioctl has stalled.

To read from a virtual address the corresponding VMID (GPU virtual address space identifier) must be known. This is a dynamic setting that can vary over time as processes are created/destroyed or when the GPU resources (VMIDs/queues) are oversubscribed. For a single process: VMID numbering begins from 8 in the current implementation.

Caution: Userptr (temporarily pinned) memory cannot be examined with this method, unless the kernel has been compiled with CONFIG_STRICT_DEVMEM=0. System memory allocations through libhsakmt default to userptr, unless the environment variable HSA_USERPTR_FOR_PAGED_MEM is set to 0.



umr -vr 8@7ffff4a01b00 40 | xxd -e
00000000: c0060080 00000000 c0020100 00000008  ................
00000010: bf8cc07f 80848104 87040404 bf85fffd  ................
00000020: 7e000202 7e020203 7e0402ff 12345678  ...~...~...~xV4.
00000030: dc700000 00000200 bf810000 bf800000  ..p.............
The output of the -vr command is the binary contents of memory at the specified base address and size (both in hex). To make it human-readable the output is piped into the xxd command.

In addition to showing raw memory UMR can disassemble memory, with the -vdis command, if the target address contains shader instructions.

umr -vdis 8@7ffff4a01b00 40
    pgm[8@0x7ffff4a01b00 + 0x0   ] = 0xc0060080         s_load_dwordx2 s[2:3], s[0:1], 0x0
    pgm[8@0x7ffff4a01b00 + 0x4   ] = 0x00000000 ;;
    pgm[8@0x7ffff4a01b00 + 0x8   ] = 0xc0020100         s_load_dword s4, s[0:1], 0x8
    pgm[8@0x7ffff4a01b00 + 0xc   ] = 0x00000008 ;;
    pgm[8@0x7ffff4a01b00 + 0x10  ] = 0xbf8cc07f         s_waitcnt lgkmcnt(0)
    pgm[8@0x7ffff4a01b00 + 0x14  ] = 0x80848104         s_sub_u32 s4, s4, 1
In some cases it is useful to see the virtual to physical mapping of a given VMID/address pair. The -vm option reveals this.

umr -vm 8@7ffff4a01b00 1
[..]
BASE=0x00000003fefee001, VA=0x000000000000, PBA==0x0003fefee000, V=1, S=0, C=0, P=0
   \-> PDE2=0x00000003fec03001, VA=0x7f8000000000, PBA==0x0003fec03000, V=1, S=0, C=0, P=0
      \-> PDE1=0x00000003fec04001, VA=0x007fc0000000, PBA==0x0003fec04000, V=1, S=0, C=0, P=0
         \-> PTE==0x0040000000e004f1, VA=0x000000001000, PBA==0x000000e00000, V=1, S=0, P=0
         \-> PTE==0x0040000000e004f1, VA=0x000000002000, PBA==0x000000e00000, V=1, S=0, P=0
Here we see a 3-level page table hierarchy. At each level the physical address of the next page directory/table is calculated. At the final level the physical address of the 2MB page is shown: 0xe00000. The offset within this page must be added manually, giving 0xe01b00. amdgpu uses both 2MB and 4KB pages as needed.

UMR can read from this address directory, by omitting the VMID@ prefix to the earlier commands. However, care must be taken not to leave the extent of the physical page.

umr -vr e01b00 40 | xxd -e
00000000: c0060080 00000000 c0020100 00000008  ................
00000010: bf8cc07f 80848104 87040404 bf85fffd  ................
00000020: 7e000202 7e020203 7e0402ff 12345678  ...~...~...~xV4.
00000030: dc700000 00000200 bf810000 bf800000  ..p.............
Caution: This method only works with VRAM physical addresses (S=0 in the PTE).

Wavefront Access
UMR can capture a snapshot of active wavefronts, including state registers and GPRs. To capture an atomic state it is necessary to halt the wavefronts, then read the state, and unhalt after. The -O halt_waves option takes care of this.

umr -O bits,halt_waves -wa
Main Registers:
      pc_hi: 00007f6c |                pc_lo: 5ce02100 |        wave_inst_dw0: bf82ffff |        wave_inst_dw1: bf810000 |
    exec_hi: 00000000 |              exec_lo: ffffffff |               tba_hi: 00000000 |               tba_lo: 00000000 |
     tma_hi: 00000000 |               tma_lo: 00000000 |                   m0: 00000004 |              ib_dbg0: 00000843 |
[..]
SGPRS:
[   0..   3] = { 00000000, 00000000, 00000000, 00000000 }
[   4..   7] = { 61a0c000, 00007f6c, 5ce00000, 00007f6c }
[   8..  11] = { 61a0c000, 00007f6c, 5ce00000, 00007f6c }
[  12..  15] = { 61a0c000, 00007f6c, 5ce00000, 00007f6c }
[..]
VGPRS:        t00      t01      t02      t03      t04      t05      t06      t07      t08      t09      t10      t11      t12      t13      t14      t15      t16      t17      t18      t19      t20      t21      t22      t23      t24      t25      t26      t27      t28      t29      t30      t31      (t32)    (t33)    (t34)    (t35)    (t36)    (t37)    (t38)    (t39)    (t40)    (t41)    (t42)    (t43)    (t44)    (t45)    (t46)    (t47)    (t48)    (t49)    (t50)    (t51)    (t52)    (t53)    (t54)    (t55)    (t56)    (t57)    (t58)    (t59)    (t60)    (t61)    (t62)    (t63)
    [  0] = { 00000000 00000001 00000002 00000003 00000004 00000005 00000006 00000007 00000008 00000009 0000000a 0000000b 0000000c 0000000d 0000000e 0000000f 00000010 00000011 00000012 00000013 00000014 00000015 00000016 00000017 00000018 00000019 0000001a 0000001b 0000001c 0000001d 0000001e 0000001f 00000020 00000021 00000022 00000023 00000024 00000025 00000026 00000027 00000028 00000029 0000002a 0000002b 0000002c 0000002d 0000002e 0000002f 00000030 00000031 00000032 00000033 00000034 00000035 00000036 00000037 00000038 00000039 0000003a 0000003b 0000003c 0000003d 0000003e 0000003f }
    [  1] = { 464c457f 40010102 00000001 00000000 00e00003 00000001 00002000 00000000 00000040 00000000 000048b0 00000000 0000002c 00380040 00400007 0009000b 00000006 00000004 00000040 00000000 00000040 00000000 00000040 00000000 00000188 00000000 00000188 00000000 00000008 00000000 00000001 00000004 00000000 00000000 00000000 00000000 00000000 00000000 000008a0 00000000 000008a0 00000000 00001000 00000000 00000001 00000005 00001000 00000000 00002000 00000000 00002000 00000000 00002c80 00000000 00002c80 00000000 00001000 00000000 00000001 00000006 00003c80 00000000 00005c80 00000000 }
    [  2] = { 61a11cf0 61a11cf1 61a11cf2 61a11cf3 61a11cf4 61a11cf5 61a11cf6 61a11cf7 61a11cf8 61a11cf9 61a11cfa 61a11cfb 61a11cfc 61a11cfd 61a11cfe 61a11cff 61a11d00 61a11d01 61a11d02 61a11d03 61a11d04 61a11d05 61a11d06 61a11d07 61a11d08 61a11d09 61a11d0a 61a11d0b 61a11d0c 61a11d0d 61a11d0e 61a11d0f 61a11d10 61a11d11 61a11d12 61a11d13 61a11d14 61a11d15 61a11d16 61a11d17 61a11d18 61a11d19 61a11d1a 61a11d1b 61a11d1c 61a11d1d 61a11d1e 61a11d1f 61a11d20 61a11d21 61a11d22 61a11d23 61a11d24 61a11d25 61a11d26 61a11d27 61a11d28 61a11d29 61a11d2a 61a11d2b 61a11d2c 61a11d2d 61a11d2e 61a11d2f }
    [  3] = { 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c 00007f6c }
[..]
PGM_MEM:
[..]
*  pgm[8@0x7f6c5ce020e0 + 0x20  ] = 0xbf82ffff         s_branch 65535
   pgm[8@0x7f6c5ce020e0 + 0x24  ] = 0xbf810000         s_endpgm
CPC Dump
A summary of key CPC registers from active HQDs is available through the -cpc option (binary build only; not yet upstreamed).

umr -cpc
 
ME 1 Pipe 0: INSTR_PTR 0x47a  INT_STAT_DEBUG 0x4000000
Pipe 0  Queue 2  VMID 8
  PQ BASE 0x7f6c61920000  RPTR 0x10  WPTR 0x10  RPTR_ADDR 0x7f6c61a04080  CNTL 0xc01c50d
  EOP BASE 0x7f6c619be000  RPTR 0x40000000  WPTR 0x3f70000  WPTR_MEM 0x8
  MQD 0xa02800  DEQ_REQ 0x0  IQ_TIMER 0x0  AQL_CONTROL 0x1
  SAVE BASE 0x0  SIZE 0x0  STACK OFFSET 0x0  SIZE 0x0
ME 1 Pipe 1: INSTR_PTR 0x47a  INT_STAT_DEBUG 0x4000000
Pipe 1  Queue 2  VMID 8
  PQ BASE 0x7f6c619c0000  RPTR 0x20  WPTR 0x20  RPTR_ADDR 0x7f6c61a32080  CNTL 0xc01c50f
  EOP BASE 0x7f6c61a23000  RPTR 0x40000010  WPTR 0x3ff8010  WPTR_MEM 0x10
  MQD 0xa01e00  DEQ_REQ 0x0  IQ_TIMER 0x0  AQL_CONTROL 0x1
  SAVE BASE 0x0  SIZE 0x0  STACK OFFSET 0x0  SIZE 0x0
ME 1 Pipe 2: INSTR_PTR 0x47a  INT_STAT_DEBUG 0x4000000
ME 1 Pipe 3: INSTR_PTR 0x47a  INT_STAT_DEBUG 0x4000000
ME 2 Pipe 0: INSTR_PTR 0x47a  INT_STAT_DEBUG 0x0
Pipe 0  Queue 0  VMID 0
  PQ BASE 0xa00000  RPTR 0x94  WPTR 0x94  RPTR_ADDR 0xa01800  CNTL 0xc0008508
  EOP BASE 0xa00800  RPTR 0x40000000  WPTR 0x3ff8000  WPTR_MEM 0x0
  MQD 0x10847dd1000  DEQ_REQ 0x0  IQ_TIMER 0x0  AQL_CONTROL 0x0
  SAVE BASE 0x0  SIZE 0x0  STACK OFFSET 0x0  SIZE 0x0
ME 2 Pipe 1: INSTR_PTR 0x47a  INT_STAT_DEBUG 0x0
Pipe 1  Queue 0  VMID 0
  PQ BASE 0x455000  RPTR 0x6b00  WPTR 0x6b00  RPTR_ADDR 0x4005c0  CNTL 0xc0308011
  EOP BASE 0x454000  RPTR 0x40000000  WPTR 0x3ff8000  WPTR_MEM 0x0
  MQD 0xf400a0a000  DEQ_REQ 0x0  IQ_TIMER 0x0  AQL_CONTROL 0x0
  SAVE BASE 0x0  SIZE 0x0  STACK OFFSET 0x0  SIZE 0x0
SDMA Dump
A summary of key SDMA registers from active HQDs is available through the -sdma option (binary build only; not yet upstreamed).

umr -sdma
 
SDMA0_STATUS2_REG: 0x930
 
SDMA0 GFX
  RB BASE 0x55d000  RPTR 0xc0  WPTR 0xc0  RPTR_ADDR 0x400660  CNTL 0x41017
  CTX STATUS  0x4
 
SDMA0 PAGE
  RB BASE 0x55f000  RPTR 0x840  WPTR 0x840  RPTR_ADDR 0x400700  CNTL 0x41025
  CTX STATUS  0xD
 
SDMA1_STATUS2_REG: 0x929
 
SDMA1 GFX
  RB BASE 0x65f000  RPTR 0xc0  WPTR 0xc0  RPTR_ADDR 0x4007a0  CNTL 0x41017
  CTX STATUS  0x4
 
SDMA1 PAGE
  RB BASE 0x661000  RPTR 0xc0  WPTR 0xc0  RPTR_ADDR 0x400840  CNTL 0x41017
  CTX STATUS  0xD
KFD Debugfs
All CP and SDMA MQDs managed by the KFD can be viewed in debugfs:

sudo cat /sys/kernel/debug/kfd/mqds
 
Process 8319 PASID 0x8001:
  Compute queue on device 1bcb
    00000000: c0310800 00000000 00000000 00000000 00000000 00000000 00000000 00000000
    00000020: 00000000 00000000 00000000 00000001 00000000 00000000 00000000 00000000
    00000040: 00000000 00000000 00000000 00000000 00000040 00000000 00000000 ffffffff
    00000060: ffffffff 00000000 ffffffff ffffffff 00000000 00000000 00000000 00000000
    00000080: 00000000 00000000 00000000 00000000 00000000 00000000 00000000 ffffffff
[..]
PM4 runlists currently executed by CP schedulers can also be viewed:

sudo cat /sys/kernel/debug/kfd/rls
 
Node 2, gpu_id 1bcb:
  00000000: c00ea100 14008001 febfe001 00000003 00010002 00000018 0000ffe0 80000000
  00000020: 0000fff0 00000000 00000000 00000000 00000000 00800080 00000000 00000000
  00000040: c005a200 20000010 00004008 00958000 00000000 f3502038 00007f08 c005a200
  00000060: 20000010 00004000 00952000 00000000 f3526038 00007f08
KFD Dynamic Debug
The driver has a number of kernel logging statements (pr_debug) which are disabled by default. These can be enabled at runtime by a script in the brahma-utils repository:

sudo kfd-tools/scripts/enable_dynamic_debug.sh -m amdgpu
HDT (Hardware Debug Tool)
Software: http://rms/downloads (Tools - Windows → Hardware Debug Tool)

Caution: HDT software installations have a short expiry and must be updated regularly, including to current alpha releases. Some ASICs require specific HDT software, found in the "SCBU Software" categories. ASIC compatibility is listed on the download page.

While many macroarchitecture registers are accessible through software, there are many more inaccessible microarchitecture registers and FIFOs. These can only be inspected via a "scan dump", using "Wombat" hardware connected to the board via JTAG. This level of detail is not needed for most debugging sessions but can be useful in some cases.

The Wombat hardware is paired with HDT software. HDT connects to the Wombat through an IP address (displayed on the hardware itself).

Once connected it is usually necessary to "unlock" the ASIC before taking a scan. This option can be found under "Security → Status and Secured Unlock".

The scan dump feature can be found under "Debug Utilities → Scan". The user must log in to the server before a scan can be taken.

Caution: The scan dump username must be prefixed with "AMD\". Multiple failed logins will lead to silent login rejection for a short period of time.

After a scan has been started, by hitting the Scan button (leave other options at defaults), the software will taken a minute or two to collect the data. It will then prompt to upload the scan to the "scanview" server and provide a URL in the output window.

The scan data is a text file, which contains millions of lines like this:

SYS.PKG00.interposer.gpu.CHIP.compute_array0.gc_ldsq_t0.sq.ib.ibuf0.simd[0].wave_info.info_reg[pc]      ffe15ch
SYS.PKG00.interposer.gpu.CHIP.compute_array0.gc_ldsq_t0.sq.ib.ibuf0.simd[0].wave_info.info_reg[pc_not_valid]    0h
SYS.PKG00.interposer.gpu.CHIP.compute_array0.gc_ldsq_t0.sq.ib.ibuf0.simd[0].wave_info.info_reg[perf_en] 0h
SYS.PKG00.interposer.gpu.CHIP.compute_array0.gc_ldsq_t0.sq.ib.ibuf0.simd[0].wave_info.info_reg[pops_packer0]    0h
SYS.PKG00.interposer.gpu.CHIP.compute_array0.gc_ldsq_t0.sq.ib.ibuf0.simd[0].wave_info.info_reg[pops_packer1]    0h
SYS.PKG00.interposer.gpu.CHIP.compute_array0.gc_ldsq_t0.sq.ib.ibuf0.simd[0].wave_info.info_reg[priv]    1h

Scanview offers a web interface to search this text. It can also email a link to the (multi-GB) text file for download and offline search.

The web interface also features a number of "reports", which are small programs to automatically extract and format information from the scan. One example is the "CPC gfx9 triage part 1" report, a summary of CPC state maintained by the CP team

Debugging Triage and Techniques
Memory
Physical Memory
The GPU has access to several kinds of memory:

VRAM (also called LFB, local framebuffer)
System memory (coherent with CPU)
I/O addresses (e.g. remote GPU framebuffer)
Each request in the memory pipeline identifies the physical address space: local or remote. This is taken from the virtual address mapping (see next section).

The overall address space has a size limit determined by the ASIC. e.g. MI100 has a 48-bit physical address space. This may become relevant when trying to access system addresses which exceed this, because the CPU in the system supports a larger address space. In these cases it can be necessary to make BIOS adjustments to limit the system physical address size.

The memory pipeline will typically traverse one or more data caches. I have not found a good overview of these caches for gfx9. Here is a useful document for gfx10. The main change since gfx9 is the addition of a read-only gL1 cache between SQC/TCP (these now called gL0) and TC (now called gL2): http://p4web.amd.com:1677/@md=d&cd=//gfxip/gfx10/doc/architecture/SystemSpecification/&cdf=//gfxip/gfx10/doc/architecture/SystemSpecification/gfx10_cache_system.docx&c=d1i@//gfxip/gfx10/doc/architecture/SystemSpecification/gfx10_cache_system.docx?ac=22 

Memory requests in ROCm originate from several clients:

From compute dispatches to TC
SQC instruction cache load
SQC scalar data cache load/store
TCP vector data cache load/store
From CPC to TC
Ring buffer load, MQD load/store, etc.
From SDMA to TC
Ring buffer load, bulk data load/store
From TC to VRAM or PCI bus
On behalf of clients above
Or by itself, through writeback
From remote clients to VRAM
Load/store through HDP cache
It is not easy to observe this transactions. There are very small SRAMs and FIFOs carrying information about in-flight transactions, which can be traced (with difficulty) in a scan dump. However, these cover only a tiny snapshot of the overall activity in a system. Any subsequent transactions quickly overwrite evidence of earlier ones.

The only way to reliably observe an in-flight transaction is when it cannot be completed (due to e.g. a hardware bug). It is still generally easier to observe the state of the client which issued the transaction and infer its details, however.

Virtual Memory
Under the hood most caches deal with physical addresses. Their clients, however, almost exclusively work with virtualized address spaces. There are two levels of address mapping:

Virtual to guest physical
Translated by VM block in UTC L2
Translations cached by UTC L2 and many UTC L1s
Guest physical to system physical
Translated by system IOMMU
Translations cached by IOMMU L1
Usually an identity mapping unless system IOMMU is configured for device isolation
The first category can be further subdivided:

GPU virtual (called GPUVM or just VM) to guest physical
Page tables located in VRAM
Target may be local or remote address
System virtual to guest physical
Page tables shared with a CPU process
Target may only be remote address
Translated by IOMMU via PCI ATS protocol
Translations cached by IOMMU L2 and UTC L2
System virtual addressing is only used by integrated GPUs (APUs). The rest of this document will focus on GPUVM addressing, the method used by discrete GPUs. Guest-system physical translation will not be covered as it rarely causes issues.

Page Tables
Top-level specs: https://amdcloud.sharepoint.com/sites/gmc/Shared%20Documents/Forms/AllItems.aspx?viewid=64fcf81e%2D1fe4%2D4d4d%2D9134%2Da616ee629d52
gfx9 spec: https://amdcloud.sharepoint.com/:w:/r/sites/gmc/Shared%20Documents/GMCv9.0%20Pirate%20Islands%20(Greenland,%20Raven1x)/GPUVM%20ATC%20L2/Greenland_VM_Programming_Guide.docx?d=w6ceb7acbdd6f4f01a093ba2932f69108&csf=1&web=1&e=npYonL
gfx10 spec: https://amdcloud.sharepoint.com/:w:/r/sites/gmc/Shared%20Documents/GMHUBv2.x/utc/NV%20GPUVM%20Programming%20Guide.docx?d=we79c3a07bd0e40b79097f3d6ff00a97e&csf=1&web=1&e=25U3Rk

There are two independent VM systems, in GFXHUB and MMHUB. The former is used by graphics clients (CP, CUs) and the latter by SDMA engines. The two systems are generally programmed identically, so it may be simpler to focus on one.

VM address spaces are identified by their VMID, numbered 0-15. Every VM client (CP, SDMA, CUs) sends a VMID with its address translation requests. Up to 16 VMIDs may be configured concurrently with independent page tables.

VMID 0 is reserved for privileged clients (CP, SDMA) controlled by the driver. VMIDs 1-15 are dynamically assigned by the scheduler, one per process. In the current driver VMIDs 1-7 are reserved for KMD clients and 8-15 are used by ROCm. This may change in the future.

VMID 0
VMID 0 uses a single page table. Each entry maps 4KB of physical address space. Here is an example translation by UMR:

umr -vm 0@b00000 1
 
=== VM Decoding of address 0@0xb00000 ===
mmVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32=0x0
mmVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32=0x0
mmVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32=0x900001
mmVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32=0x0
mmVM_CONTEXT0_CNTL=0x7ffe01
VMID0.page_table_block_size=0
VMID0.page_table_depth=0
mmVGA_MEMORY_BASE_ADDRESS=0x0
mmVGA_MEMORY_BASE_ADDRESS_HIGH=0x0
mmMC_VM_FB_OFFSET=0x0
mmMC_VM_MX_L1_TLB_CNTL=0x3859
mmMC_VM_SYSTEM_APERTURE_LOW_ADDR=0x3d0000
mmMC_VM_SYSTEM_APERTURE_HIGH_ADDR=0x3fffffff
mmMC_VM_FB_LOCATION_BASE=0xf400
mmMC_VM_FB_LOCATION_TOP=0xf7fe
mmMC_VM_AGP_BASE=0x0
mmMC_VM_AGP_BOT=0x0
mmMC_VM_AGP_TOP=0x0
PDE=0x0000000000900001, PBA==0x000000900000, V=1, S=0, FS=0
\-> PTE=0x0600001044400073, VA=0x0000000000b00000, PBA==0x001044400000, F=0, V=1, S=1
   \-> Computed address we will read from: sys:1044400000 (reading: 4096 bytes)
 
=== Completed VM Decoding ===
The page table base address is programmed in VM_CONTEXT0_PAGE_TABLE_BASE_ADDR. The low 12 bits are discarded, giving physical address 0x900000.

Caution: There are multiple VMs which must always be programmed together. 1 VM in GFXHUB and 1 VM per MMHUB.

The virtual address space may start from a non-zero value. VM_CONTEXT0_PAGE_TABLE_START_ADDR records the starting virtual address. It is 0 in this case.

We can perform this translation manually. Since each entry maps 4KB the entry for virtual address 0xb00000 will be at index 0xb00. Each entry is 8 bytes, so we should find the entry at 0x900000 + ((0xb00000 / 0x1000) * 0x8) = 0x905800:

umr -vr 905800 8 | xxd -e
00000000: 44400073 06000010                    s.@D....
The PTE is 0x600001044400073, matching the one found by UMR. The meaning of the non-address bits can be found in the specifications linked earlier. UMR also decodes a few: V = valid, S = system (0 meaning VRAM).

The page starts at system physical address 0x1044400000.

Apertures
Although the page table structure in VMID 0 is simple, there are three apertures inside which the page table is not used:

mmMC_VM_SYSTEM_APERTURE_LOW_ADDR=0x3d0000
mmMC_VM_SYSTEM_APERTURE_HIGH_ADDR=0x3fffffff
mmMC_VM_FB_LOCATION_BASE=0xf400
mmMC_VM_FB_LOCATION_TOP=0xf7fe
mmMC_VM_AGP_BOT=0xf800
mmMC_VM_AGP_TOP=0xffffff
The system aperture (0xf4'0000'0000-0xffff'ffff'ffff) defines a region inside which virtual:physical address mappings are linear (identity with a constant offset). It encompasses both of the other apertures.

The framebuffer aperture (0xf4'0000'0000-0xf7'feff'ffff) is a linear mapping of VRAM. To find the VRAM physical address subtract the base address of the aperture.

The AGP aperture (0xf8'0000'0000-0xffff'ffff'ffff) is a linear mapping of system memory. To find the system physical address subtract the base address of the aperture.

UMR does not currently handle these apertures in address queries or memory access, so you will need to do this by hand. Here are two examples:

umr -cpc
[..]
MQD 0x1084544b000  DEQ_REQ 0x0  IQ_TIMER 0x0  AQL_CONTROL 0x0
[..]
This address falls inside the AGP aperture, giving system physical address (0x1084544b000 - 0xf800000000) = 0x104544b000. UMR does not provide an interface to read system physical memory directly. Here I use a tool kmemspy to do this:

./kmemspy --phys 104544b000 10
0x104544b000:       0xc0310800      0x00000000      0x00000000      0x00000000
Here's a case with the framebuffer aperture. UMR can read physical VRAM directly so it's easy to reach the underlying data:

umr -cpc
[..]
MQD 0xf400a0a000  DEQ_REQ 0x0  IQ_TIMER 0x0  AQL_CONTROL 0x0
[..]
 
umr -vr a0a000 10 | xxd -e
00000000: c0310800 00000000 00000000 00000000  ..1.............
VMIDs 1-15
VMIDs 1-15 are assigned dynamically to per-process page tables. In ROCm there is a 1:1 mapping of VMIDs to processes. When the number of processes exceeds the number of VMIDs the scheduler (discussed in a later section) will assign subsets of processes to the available VMIDs in different timeslices.

The page tables used in these VMIDs are multi-level. For example:

umr -vm 8@0x7ffff7f76000 1
 
=== VM Decoding of address 8@0x7ffff7f76000 ===
mmVM_CONTEXT8_PAGE_TABLE_START_ADDR_LO32=0x0
mmVM_CONTEXT8_PAGE_TABLE_START_ADDR_HI32=0x0
mmVM_CONTEXT8_PAGE_TABLE_BASE_ADDR_LO32=0xfebfe001
mmVM_CONTEXT8_PAGE_TABLE_BASE_ADDR_HI32=0x3
mmVM_CONTEXT8_CNTL=0x7ffe07
VMID8.page_table_block_size=0
VMID8.page_table_depth=3
mmVGA_MEMORY_BASE_ADDRESS=0x0
mmVGA_MEMORY_BASE_ADDRESS_HIGH=0x0
mmMC_VM_FB_OFFSET=0x0
mmMC_VM_MX_L1_TLB_CNTL=0x0
mmMC_VM_SYSTEM_APERTURE_LOW_ADDR=0x0
mmMC_VM_SYSTEM_APERTURE_HIGH_ADDR=0x0
mmMC_VM_FB_LOCATION_BASE=0xf400
mmMC_VM_FB_LOCATION_TOP=0xf7fe
mmMC_VM_AGP_BASE=0x0
mmMC_VM_AGP_BOT=0x0
mmMC_VM_AGP_TOP=0x0
BASE=0x00000003febfe001, VA=0x000000000000, PBA==0x0003febfe000, V=1, S=0, C=0, P=0
   \-> PDE2@{0x3febfe7f8/ff}=0x0000000000cf1001, VA=0x7f8000000000, PBA==0x000000cf1000, V=1, S=0, C=0, P=0, FS=0
      \-> PDE1@{0xcf1ff8/1ff}=0x0000000000cf2001, VA=0x007fc0000000, PBA==0x000000cf2000, V=1, S=0, C=0, P=0, FS=0
         \-> PDE0@{0xcf2df8/1bf}=0x0000000000cf3001, VA=0x000037e00000, PBA==0x000000cf3000, V=1, S=0, C=0, P=0, FS=0
            \-> PTE@{0xcf3bb0/176}==0x00000003febfc071, VA=0x000000176000, PBA==0x0003febfc000, V=1, S=0, P=0, FS=0, F=0
               \-> Computed address we will read from: vram:3febfc000 (reading: 4096 bytes)
 
=== Completed VM Decoding ===
There are 3 page directories and 1 page table in this hierarchy. The top-level page directory starts at physical VRAM address 0x3febfe000, with 512 entries of 8 bytes each. This address comes from VM_CONTEXT8_PAGE_TABLE_BASE_ADDR.

The offset into each page directory/table are calculated by segmenting the virtual address into different pieces at each level. The translation above can be performed manually like this:

Offset into PD2
Bits 47:39 of 0x7ffff7f76000 = 0xff
PDE2 at (0x3febfe000 + (0xff * 8)) = 0x3febfe7f8
PDE2 = 0xcf1001
Offset into PD1
Bits 38:30 of 0x7ffff7f76000 = 0x1ff
PDE1 at (0xcf1000 + (0x1ff * 8)) = 0xcf1ff8
PDE1 = 0xcf2001
Offset into PD0
Bits 29:21 of 0x7ffff7f76000 = 0x1bf
PDE0 at (0xcf2000 + (0x1bf * 8)) = 0xcf2df8
PDE0 = 0xcf3001
Offset into PT
Bits 20:12 of 0x7ffff7f76000 = 0x176
PTE at (0xcf3000 + (0x176 * 8)) = 0xcf3bb0
PTE = 0x3febfc071
Address of page: 0x3febfc000
Large Pages
A PDE may behave as a PTE if bit 54 is set. The driver sets this bit in PD0 to map a 2MB physical VRAM or system memory page. Translations for large pages are cached more efficiently in UTCs. Here's an example:

umr -vm 8@0x7ffff4a00000 1
 
=== VM Decoding of address 8@0x7ffff4a00000 ===
mmVM_CONTEXT8_PAGE_TABLE_START_ADDR_LO32=0x0
mmVM_CONTEXT8_PAGE_TABLE_START_ADDR_HI32=0x0
mmVM_CONTEXT8_PAGE_TABLE_BASE_ADDR_LO32=0xfebfe001
mmVM_CONTEXT8_PAGE_TABLE_BASE_ADDR_HI32=0x3
mmVM_CONTEXT8_CNTL=0x7ffe07
VMID8.page_table_block_size=0
VMID8.page_table_depth=3
mmVGA_MEMORY_BASE_ADDRESS=0x0
mmVGA_MEMORY_BASE_ADDRESS_HIGH=0x0
mmMC_VM_FB_OFFSET=0x0
mmMC_VM_MX_L1_TLB_CNTL=0x0
mmMC_VM_SYSTEM_APERTURE_LOW_ADDR=0x0
mmMC_VM_SYSTEM_APERTURE_HIGH_ADDR=0x0
mmMC_VM_FB_LOCATION_BASE=0xf400
mmMC_VM_FB_LOCATION_TOP=0xf7fe
mmMC_VM_AGP_BASE=0x0
mmMC_VM_AGP_BOT=0x0
mmMC_VM_AGP_TOP=0x0
BASE=0x00000003febfe001, VA=0x000000000000, PBA==0x0003febfe000, V=1, S=0, C=0, P=0
   \-> PDE2@{0x3febfe7f8/ff}=0x0000000000cf1001, VA=0x7f8000000000, PBA==0x000000cf1000, V=1, S=0, C=0, P=0, FS=0
      \-> PDE1@{0xcf1ff8/1ff}=0x0000000000cf2001, VA=0x007fc0000000, PBA==0x000000cf2000, V=1, S=0, C=0, P=0, FS=0
         \-> PTE@{0xcf2d28/0}==0x06400007ed2004f7, VA=0x000000000000, PBA==0x0007ed200000, V=1, S=1, P=0, FS=9, F=0
            \-> Computed address we will read from: sys:7ed200000 (reading: 4096 bytes)
 
=== Completed VM Decoding ===
Notice the absence of PDE0 in the page table walk above. It has instead been interpreted as a PTE with coverage of 2MB.

It is also possible to map 1GB pages in PD1, but the driver does not currently do this.

Address Translation Faults
A translation fault arises when a virtual address does not have a valid mapping in the page table (V=0) or the requested permissions do not match (e.g. write request to a page with W=0). There are two kinds of faults:

Non-recoverable fault: an error which should result in program termination or debugger attach
Analogous to a CPU segmentation fault
Recoverable fault: a temporary state which notifies the driver to populate the page table entry
The driver configuration determines the type of fault. Legacy mode (amdgpu.noretry=1) requires all pages to have valid PTEs before UTC clients attempt to access them. In this mode all faults are errors.

A newer mode (amdgpu.noretry=0) makes use of recoverable faults. All pages are physically backed by system memory (or disk, or other faultable source) and moved in/out of VRAM as needed. Recoverable faults allow the driver to stall access to a page while it's being moved.

There may also be invalid translations in the newer mode. In this case a recoverable fault will be changed to a non-recoverable fault by the driver (by manipulating the PTE), allowing error handling to behave the same as in legacy mode.

Interrupt and Logging
When the page table walker determines that a translation should result in a non-recoverable fault, it may also be configured to notify the driver (VM_L2_PROTECTION_FAULT_CNTL.*_INTERRUPT=1). The driver will print details of the fault to the kernel log and terminate the originating process, or allow the debugger to attach.

Fault logging looks like this:

amdgpu 0000:84:00.0: amdgpu: [gfxhub0] no-retry page fault (src_id:0 ring:40 vmid:8 pasid:32769, for process hsatest pid 3148 thread hsatest pid 3148)
amdgpu 0000:84:00.0: amdgpu:   in page starting at address 0x0000001234567000 from client 27
amdgpu 0000:84:00.0: amdgpu: VM_L2_PROTECTION_FAULT_STATUS:0x00841050
amdgpu 0000:84:00.0: amdgpu:         Faulty UTCL2 client ID: 0x8
amdgpu 0000:84:00.0: amdgpu:         MORE_FAULTS: 0x0
amdgpu 0000:84:00.0: amdgpu:         WALKER_ERROR: 0x0
amdgpu 0000:84:00.0: amdgpu:         PERMISSION_FAULTS: 0x5
amdgpu 0000:84:00.0: amdgpu:         MAPPING_ERROR: 0x0
amdgpu 0000:84:00.0: amdgpu:         RW: 0x1
The PCI address of the GPU is shown on every line. This fault originated from GFXHUB. The VMID and PASID are shown: 8 and 32769 respectively. Details of the process corresponding to this PASID are also shown.

The beginning virtual address of the page is shown: 0x1234567000. The low 12 bits will always be zero. It is not possible to determine the complete address from this information.

The UTC L2 client ID helps to determine the source of the fault. These are listed in the VM programming guide linked at the top of this section. 8 corresponds to the TCP, a write-through vector cache used by compute units. We can determine that this fault was triggered by a shader.

Some additional information about the permissions are provided. This translation was a write request (bit 2 in PERMISSION_FAULTS and the RW flag) for a valid entry (bit 0 in PERMISSION_FAULTS). These flags must be compatible with those in the PTE to avoid a fault.

Fault vs Aperture Violation
A VM fault will only occur if the virtual address is within hardware limits. For current ASICs this is < 48 bits.

A translation request for an address beyond this limit will not be reported in the same way. It is categorized as a device unified address (DUA) aperture violation. This will not be reported to, or logged by, the driver.

An aperture violation is handled differently depending on the client:

The CP will set the flag SUA_VIOLATION_INT_STATUS  in the CP_MEx_INT_STAT_DEBUG register
The SQ will raise a memory violation exception, transferring control flow to the trap handler
The trap handler will halt the wave and report the error to ROCr
Caching
Data Caching
In addition to defining address translations and access permissions the PTE defines cache behavior for clients reading/writing the page. This field is called MTYPE (M in the VM programming guide).

Address Translation Caching
CP
Overview
gfx9 PM4 spec: http://p4web.amd.com:1677/@md=d&cd=//gfxip/gfx9/doc/design/blocks/cp/packets/greenland/&cdf=//gfxip/gfx9/doc/design/blocks/cp/packets/greenland/cp_packets_gr.pdf&sr=4490385&c=sxq@//gfxip/gfx9/doc/design/blocks/cp/packets/greenland/cp_packets_gr.pdf
gfx10 PM4 spec: http://p4web.amd.com:1677/@md=d&cd=//gfxip/gfx10/doc/blocks/cp/packets/navi10/&cdf=//gfxip/gfx10/doc/blocks/cp/packets/navi10/cp_packets_nv10.pdf&sr=4490385&c=EfU@//gfxip/gfx10/doc/blocks/cp/packets/navi10/cp_packets_nv10.pdf
AQL spec: http://www.hsafoundation.com/?ddownload=5702

Communication with the CP takes place through queues. A queue has a ring buffer base address, size, read/write pointers, and other properties. For example:

umr -cpc
 
ME 1 Pipe 0: INSTR_PTR 0x10f7  INT_STAT_DEBUG 0x4000000
Pipe 0  Queue 2  VMID 8
  PQ BASE 0x7f08f3420000  RPTR 0x0  WPTR 0x10  RPTR_ADDR 0x7f08f3502080  CNTL 0xc01450d
  EOP BASE 0x7f08f34b4000  RPTR 0x40000000  WPTR 0x3ff8000  WPTR_MEM 0x0
  MQD 0x958000  DEQ_REQ 0x0  IQ_TIMER 0x0  AQL_CONTROL 0x1
  SAVE BASE 0x7f07e9400000  SIZE 0x1555000  STACK OFFSET 0x5000  SIZE 0x5000
ME 1 Pipe 1: INSTR_PTR 0x47a  INT_STAT_DEBUG 0x4000000
Pipe 1  Queue 2  VMID 8
  PQ BASE 0x7f08f34c0000  RPTR 0x20  WPTR 0x20  RPTR_ADDR 0x7f08f3526080  CNTL 0xc01c50f
  EOP BASE 0x7f08f3519000  RPTR 0x40000000  WPTR 0x3ff8000  WPTR_MEM 0x0
  MQD 0x952000  DEQ_REQ 0x0  IQ_TIMER 0x0  AQL_CONTROL 0x1
  SAVE BASE 0x7f07eaa00000  SIZE 0x1555000  STACK OFFSET 0x5000  SIZE 0x5000
ME 1 Pipe 2: INSTR_PTR 0x47a  INT_STAT_DEBUG 0x4000000
ME 1 Pipe 3: INSTR_PTR 0x47a  INT_STAT_DEBUG 0x4000000
ME 2 Pipe 0: INSTR_PTR 0x47a  INT_STAT_DEBUG 0x0
Pipe 0  Queue 0  VMID 0
  PQ BASE 0xb00000  RPTR 0x1c  WPTR 0xa53da1c  RPTR_ADDR 0xb01800  CNTL 0xc0008508
  EOP BASE 0xb00800  RPTR 0x40000000  WPTR 0x3ff8000  WPTR_MEM 0x0
  MQD 0x10843fc0000  DEQ_REQ 0x0  IQ_TIMER 0x0  AQL_CONTROL 0x0
  SAVE BASE 0x0  SIZE 0x0  STACK OFFSET 0x0  SIZE 0x0
ME 2 Pipe 1: INSTR_PTR 0x47a  INT_STAT_DEBUG 0x0
Pipe 1  Queue 0  VMID 0
  PQ BASE 0x456000  RPTR 0x33700  WPTR 0x1f633700  RPTR_ADDR 0x4005a0  CNTL 0xc0308011
  EOP BASE 0x455000  RPTR 0x40000000  WPTR 0x3ff8000  WPTR_MEM 0x0
  MQD 0xf40010a000  DEQ_REQ 0x0  IQ_TIMER 0x0  AQL_CONTROL 0x0
  SAVE BASE 0x0  SIZE 0x0  STACK OFFSET 0x0  SIZE 0x0
Each indented block identifies a queue. There are four queues in this example. Take this queue as an example:

ME 1 Pipe 0: INSTR_PTR 0x10f7  INT_STAT_DEBUG 0x4000000
Pipe 0  Queue 2  VMID 8
The queue is managed by ME 1, also known as MEC 1. There are two MECs in most ASICs.
It is running on pipe 0. Each MEC has 4 pipes. Pipes are independent threads of execution.
Pipe 0 is currently running the microcode instruction at offset 0x10f7. (We'll explore this later.)
A timestamp interrupt is currently raised by this pipe (this can be ignored).
The queue is running on slot 2. Each pipe has 8 queue slots (fewer in gfx10+). Slots 0-1 are currently reserved for graphics.
Only one queue is actively processed at a time. Queues are round-robin scheduled within the pipe.
All virtual memory associated with the queue is in VMID 8.
PQ BASE 0x7f08f3420000  RPTR 0x0  WPTR 0x8010  RPTR_ADDR 0x7f08f3502080  CNTL 0xc01450d
The PQ (primary queue) ring buffer base address is 0x7f08f3420000.
The read pointer is at offset 0x0 dwords.
The write pointer is 0x8010 dwords.
This value is monotonic. It must be masked by the queue size to find the offset into the ring.
The CNTL field is CP_HQD_PQ_CONTROL:
QUEUE_SIZE	5:0	0x9	Size of the primary queue (PQ) will be: 2^(HQD_QUEUE_SIZE+1) DWs. Min Size is 7 (2^8 = 256 DWs) and max size is 29 (2^30 = 1 G-DW). Values outside of this range will clamp to the min or max accordingly. The size will be used when to wrap in the primary queue for both the fetching and the reporting of the read offset.

The ring size is 2^(0xd+1) dwords = 0x4000 dwords.
The write pointer into the ring is (0x8010 & 0x3fff) = 0x10 dwords = 0x40 bytes.
The commands between the read and write pointers have not been fully processed.
Hardware Queue Descriptor (HQD)
All properties of a scheduled queue are recorded in the HQD. The HQD is a set of registers instanced per queue slot per pipe. These can be read like any other register, with the ME/pipe/queue index register set appropriately. For example, the ring base address for the queue above can be read directly:

umr -sb 1 0 2 -r vega10.gfx90.mmCP_HQD_PQ_BASE
0x08f34200
umr -sb 1 0 2 -r vega10.gfx90.mmCP_HQD_PQ_BASE_HI
0x0000007f
HQD registers follow the naming scheme CP_HQD_* and their definitions can be found in the register specifications.

umr -lr vega10.gfx90 | grep CP_HQD
        vega10.gfx90.mmCP_HQD_GFX_CONTROL (0) => 0x0323e
        vega10.gfx90.mmCP_HQD_GFX_STATUS (0) => 0x0323f
        vega10.gfx90.mmCP_HQD_ACTIVE (0) => 0x03247
        vega10.gfx90.mmCP_HQD_VMID (0) => 0x03248
[..]
HQD registers are programmed by the scheduler. This is typically a microcode program running on MEC 2 but can also be a software implementation in the driver.

Memory Queue Descriptor (MQD)
The MQD is memory storage for the contents of an HQD and other per-queue registers. This is used to provide initial register settings to the scheduler. It also records HQD settings when a queue is descheduled. The contents of the MQD are defined in the appendix of the PM4 command specification.

From the queue earlier:

MQD 0x958000  DEQ_REQ 0x0  IQ_TIMER 0x0  AQL_CONTROL 0x1
We can see e.g. the PQ base address at offset 0x220. Note: MQDs are privileged data structures in VMID 0.

umr -vr 0@958000 800 | xxd -e
[..]
00000220: 08f34200 0000007f 00000000 f3502080  .B........... P.
[..]
Keep in mind that the MQD does not reflect up-to-date state until the queue has been descheduled.

An exception to this is the ADC registers (COMPUTE_*). These are saved to the MQD when a different queue on the pipe is selected to run.

All MQDs managed by the driver, whether scheduled or not, can be read from debugfs.

Privileged CP Interface (HIQ, KIQ)
Process/Queue Scheduler (HWS)
User-Mode Compute Queues
Dispatch Lifecycle
User-Mode SDMA Queues
