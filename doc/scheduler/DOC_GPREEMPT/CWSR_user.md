
# contents are from official wiki

CWSR stands for Compute Wave Save Restore. It is a mechanism used in GPU computing to manage the execution of compute waves, which are groups of threads. CWSR allows the GPU to interrupt a running compute wave, save its complete state to memory, and later restore it to continue execution. This is similar to context switching in CPU operating systems but operates at the wave (thread group) level 
1
.

Key Features of CWSR:
Preemption: Allows higher-priority work to interrupt lower-priority compute tasks.

Time-Slicing: Shares GPU resources among multiple processes.

Debugging: Enables inspection and modification of wave state during execution.

Checkpoint/Restart: Saves application state for fault tolerance.

Power Management: Saves state before entering low-power modes.

Components Saved in CWSR:
VGPRs (Vector General Purpose Registers)

SGPRs (Scalar General Purpose Registers)

Hardware Registers: Includes PC, EXEC, STATUS, MODE, etc.

LDS (Local Data Share): Shared memory for the workgroup.

ACC VGPRs: Matrix/AI acceleration registers (GFX9+).

Shared VGPRs: Shared between waves (GFX10+)

## Section 2
Compute wave front save & restore (CWSR) is a Gfx8 feature.
CWSR provides a finer level of preemption support at the wave front level, compared to the current mid-buffer level.
CWSR requires the primary ring’s VMID the same as indirect buffer’s VMID, which means it has to run with HWS.

### 2.1 Preemption with HWS
Driver preemption sequence (preemption happens on OS queue):
Stop the run list
KMD issues UNMAP to un-map all active user queues.	
For user queues associated to the preempted OS queue, clear the “enable_queue” flag in the user queues’ MQD so that scheduler doesn’t map the queue even if it’s in the runlist
KMD reports back the preemption fence to OS
Start the runlist
Resubmission:
When a preempted IB is resubmitted, KMD doesn’t need to submit anything to the user queue as the scheduler saves the h/w states at the preemption time 
KMD needs to patch the OS fence with the new one which comes with the resubmission 
KMD needs to ring the doorbell for the resubmission with the doorbell value corresponding to the resubmitted IB
OS Key
\CurrentControlSet\Control\GraphicsDrivers\Scheduler\EnablePreemption = 0
 
	RUN_LIST(OLD)  UNMAP_QUEUES  RUN_LIST(NEW)

### 2.2 CWSR Functionality
Enables preemption at wave front level (shader)
Triggered by following scheduler events:
Quantum expiration
Queue priority change
UNMAP_QUEUE packet from driver
Actual save/restore is transparent to SW
Can be enabled per process/queue base


### 2.3 CWSR : Driver Responsibility
Provide CWSR save area for each user queue
KMD calculates size of save area
UMD allocates the memory
Enable CWSR by programing corresponding MQD fields 
Set up trap handler
Program a few configuration registers in MQD

* CWSR : Enable CWSR
Set up trap Handler
Shader program which performs save/retore when CWSR is triggered
KMD allocates a buffer that holds the trap handler and maps the trap handler into private reserved VA space
KMD provides trap handler  buffer GPUVA (SQ_Shader_Tba_Lo/Hi) in MAP_PROCESS packet
Programing corresponding MQD fields 
CP_HQD_PERSISTENT_STATE.QSWITCH_MODE = 1
CP_HQD_PERSISTENT_STATE.RELAUNCH_WAVES = 0
CP_HQD_CTX_SAVE_BASE_ADDR_LO /HI
CP_HQD_CTX_SAVE_CONTROL
CP_HQD_CTX_SAVE_SIZE
CP_HQD_CNTL_STACK_OFFSET
CP_HQD_CNTL_STACK_SIZE

# Wiki2: https://amd.atlassian.net/wiki/spaces/BLTZ/pages/534709177/CWSR+support+on+KFD （year 2015）
Overview
CWSR(compute wave save restore) support is essential for the KFD implementation of HSA as it prevents Long run wave (> preempt latency)  to stall the HQD and  eventually halt the CP .  CWSR requires the use of a privileged shader called the CWSR trap handler that saves and restores some of the state of the shader that is being context-switched.  In this design document, we only focus on software implementation for this feature , For the hardware details please refer the reference.

Reference :
              //gfxip/gfx8/doc/design/arch/context_switch/contex_switch.docx

1.    Phase 1 implementation 
1.1 Role of CP
CP is the core of whole CWSR feature , but this document mainly describe the software stack for this feature, so from software point of view , the role for CP is simple, provide the MEC firmware  that enable the CWSR feature and provide support for software team when request.  Also the trap handler source code  is provide by CP or hardware team .
1.2 Role of run time  team
Provide the  user level trap handler source code .
Call Thunk to set up user level Trap handler.
Provide test application to verify the  CWSR feature works normally on AQL code path.
1.3 Role of Thunk
Create new interface for user to setup user level trap handler. 
hsaKmtSetTrapHandler(
    HSAuint32           NodeId,                   //IN
    void*               TrapHandlerBaseAddress,   //IN
    HSAuint64           TrapHandlerSizeInBytes,   //IN
    void*               TrapBufferBaseAddress,    //IN
    HSAuint64           TrapBufferSizeInBytes,    //IN
    );
Allocate context save and restore area(CSA, including Relaunch stack and Workgroup data) for each queue and pass them to kernel through create queue IOCTL.
1.4 Role of KFD
Memory for TBA, TMA allocation and mapping
          KFD will allocate two reserved physical  pages of memory during device init period.  One page for trap handler code(ISA) itself addressed by HW  as TBA  and one page for parameter used inside trap handler addressed by HW as TMA.   KFD fill the trap handler code during  initialization. The ISA binary itself can be put in a static array .

          KFD need to map the reserved  ISA memory into user’s address space for each process.  The TBA for each MQD belongs to the same process will be initialized with this mapped user address and TMA is set to zero   by default.

Install the first level  trap handler
          KFD will install the  trap handler by program the CP_HQD and compute related registers through  MQD  as follows

          CP_HQF_PERSISTENT_STATE.QSWITCH_MODE = 1

          COMPUTE_PGM_RSRC2.TRAP_PRESENT = 1

          Program CP_HQD_CTX_SAVE_BASE_ADDR,  CP_HQD_CTX_SAVE_CONTROL, CP_HQD_CTX_SAVE_SIZE, CP_HQD_CNTL_STACK_SIZE, CP_HQD_CNTL_STACK_OFFSET, CP_HQD_WG_STATE_OFFSET, COMPUTE_TBA_ADDR, COMPUTE_TMA_ADDR according to the mapped TBA ,TMA address  and CSA information passed from thunk.

          Reference  section 2.5 and 2.5.1  in the “context_switch.doc “  for detail programming .

1.5  Enable the  Wave front  Context Switching with hardware scheduler disabled
 With HWS disabled, KFD driver decided when and which MQD will be selected to load to the HQD

In current code, KFD will check the available HQD during queue creation period, if all HQD are been used,  the creation of queue will failed. User need to wait until there is free one if they want to continue. This existing mechanism won’t work for CWSR.  There are few major issues to enable CWSR in this mode.

. When there is no HQD is available, KFD need to decide which queue should be preempted,   It could be complicated to be fair. (Add  timer etc)

. After select victim,   KFD need to have another logic to maintain the victims and decide when and which one will be select to acquire the HQD again. This could also be complicated.

(TBD)  We plan to ignore this step and directly go with HWS enabled mode.

1.6   Enable the  Wave front Context Switching with hardware scheduler enabled
With HWS enabled,   the CWSR will be triggered in following two ways.

 The queue’s quantum is expired, HWS trigger the CWSR.  In this case , KFD driver will not aware of this
  Any queue operation (queue create, destroy and event update) that cause the change of runlist.  In this case, KFD driver need to make sure all the active queue should be preempted before send new run list. KFD will send UNMAP_QUEUE with preempt action to trigger the CWSR.
1.7  Validation of CWSR  feature
Current implementation will only be validated  on AQL code path .

Validation of the corner case fix
            Without the CWSR, the preempt wave may failed due to the  limited UNMAP latency, and will cause hang if new runlist is submit to HW scheduler in this state. This can be easily reproduced to run multiple process in an infinit repeat loop  as follows .

      sudo taskset –c 3 ./hst_perf –gtest_filter=HsaFeature.Cumasking  -- This test invoke a  long-run wave that will last about 6-7 seconds.

      open another remote console , run ./kfdtest   --gtest_filter=KFDMemoryTest.BaseMemoryCopy –gtest_repeat=-1   This test case will create the queue and invoke the shader do a simple ATC memory to memory copy and exit .  The gtest_repeat =-1 will make this test repeat infiniti . 

Without CWSR , the above test will hang both test , with CWSR , we should expect both test runs normally.

AQL validation using test application provided by runtime
          Runtime team provide another test application : \\nasfs12\hsadata\public\Fan's_dropbox\Ctrl+C

          This test application (cu_mask) will first create one queue to run a long run dispatch within multiple work items (256 work group size , and 256 work group number) , each work item will loop an iterate  number and then write the  iterate number to different destination(offset based on work item ID). Before the first queue finish it's job,  another queue is  created and will  run the same  long run dispatch as the  first queue. Application will check all the  destination (array for multiple work items)  should filled with same iterate number to proved CWSR save and restored correctly for each work item . 

          Developer can run multiple instance of above test(cu_mask)  at the same time . All instance should pass the  test.

1.8  Multi level trap handler support
Multi level trap handler support depends on first level trap handler works normally.  

KFD  CWSR handler should include linkage code for jump to L2 handler, using L1 TMA to store address of L2 handler or zero if no L2 handler.

Reference “CWSR Trap handler in KFD”  for more details on multi-level trap handler support . 

The HSA runtime library need to allocate the memory(TBA) and fill in the Trap handler ISA , allocates the trap buffer(TMA) to be used for all queues associated with the specified NodeId within this process context.  HSA need to call new interface hsaKmtSetTrapHandler to kernel to install the user level trap handler .  After user call this interface, the second level trap handler for this process will be installed. (Thangirala Hari  will provide the details for the TMA layout)

2.    Phase 2 implementation
 

From phase 1 implementation, the context save & restore area is allocate and accessed by user level library. This causes security concerns since a malicious application could hang the  whole system by poison control stack data  (or even set a malicious one) thereby jeopardizing the entire system. In phase 2, CP microcode provide a workaround  that it will save a copy  of the relaunch control stack in kernel space and later SR(HW save & restore) block can restore from this copy.  This approach still leaves us a small window during which the relaunch control stack can be attacked(After SR block start the save and before  CP finished the copy) , but it is better than no protection at all.

2.1  Context save & restore area 
For each wavefront  there are two block of memory been saved & restored  differently.

 Relaunch Stack(RS)  - a control structure with a fixed-size entry per workgroup, This is written by SR block and is managed as a stack. It is written newest wave first to oldest wave lase and from high addresses to low address ed.

Workwroup Data(WD) – a large data structure containing all of the wavefront data for each wave in the workgroup.  This is  written  by each wave (trap handler)  in the  SQ.

Limited by HW design(Only have one base address register for the entire context save memory and one size register for the start of both RS and WD), both memory data structure are likely allocated as one  unit , with the workgroup data being at a fixed offset from the relaunch stack start.  Following is the layout of the whole queue context save memory .

 



 

 

2.1.1     The workaround provided from CP microcode.


The above figure shows the workaround for Carrizo.  There are enough empty spaces  in MQD  for RS block on CZ, so driver don’t need to allocate extra pages as the destination of CP ucode  DMA copy.  For other asics(ex. Fiji) . the source and destination address should be calculated as follows:

Source :   MQD_BASE_ADDR + 4k(MQD itself) + PAGE_ALIGN(RS block size)

Destination :  MQD_BASE_ADDR + 4k(MQD itself) 

 

2.1.2     Double mapping of RS block pages
From the proposal provided from CP ucode team ,  driver need to do the  double mapping of the RS block pages to dedicated  user space address(start of queue context save address)  and dedicated kernel GPU VM  address (  MQD_BASE_ADDR + 4k(MQD itself) + PAGE_ALIGN(RS block size) ). Following is a approach for the address requirement.

1.  Allcoate the MQD with extra pages including the pages will be double mapped. (Only one extra page for CZ)

2.  Runtime (user level) need to call  mmap with MAP_ANONYMOUS flag , this will not allocate any real pages, just get a continuous virtual  address.

3.  Runtime need to call KFD to MMAP with the size page aligned RS block and requested address ,  in KFD kernel ,  driver will  map the address with  the reserved pages.

4. Runtime need to call mlock on reset  save area so they would be paged out. (Not sure if this is necessary, TBD)

 # wiki3 Long Running Compute WAs
https://amd.atlassian.net/wiki/spaces/SWKMD/pages/652661086/Long+Running+Compute+WAs (year 2025)
Purpose
We have observed scenarios where some long running compute workloads can cause various TDR signatures. One issue is that our QUERY_STATUS packet cannot differentiate between some hang scenarios vs long running scenarios which forces us to implement some "hacky" workarounds based on time thresholds.

Another issue is that these long running submissions can use 100% of the CUs, which results in delayed scheduling operations from KIQ/MES as well as affecting GFX QoS. These delays result in GFX TDRs as the OS will see that some GFX submissions are not completing in time.

This document describes each WA in detail and what specific scenario they address.

Compute Preemption as Hang Detection
Previously. we would use the QUERY_STATUS packet to determine when a compute queue is hung. We do this by submitting two QUERY_STATUS packets with some time delay between them. Each packet returns some information about that queue's current state (such as rptr/wptr info). MES will compare the output of both packets and determine if that queue is idle/busy/hung. However, we have observed that the QUERY_STATUS packet cannot always determine if a long running submission is making progress. To address this, we have changed how we think about compute hangs. Instead of looking for hang signatures, the driver tries to determine if the compute engine can be preempted. If it can be preempted, then GFX will not be affected, and the user experience will not be impacted. This means that because some hang signatures are preemptible, we may allow some hangs to exist on the system. We see this as being acceptable behaviour because the system will still be usable, and the user can still kill the offending compute process if needed.

This solution utilizes the UNMAP_QUEUES packet with the unmap_all_non_static_queues bit set. This packet will unmap all compute queues (both kernel and user queues). If unmap is successful, then MES will report that compute is responsive, and remap any unmapped kernel queues. However, if unmap times out for any queue, then MES will report that compute is hung and pause scheduling. This will result in the OS to TDR and the compute engine reset will occur.

It is important to note that we only use this solution for CWSR enabled ASICs. If CWSR is not enabled, then we use the previous QUERY_STATUS solution.

Compute Submission Causing GFX Unmap Timeouts
With particular apps that submit long running compute workloads (examples: LM Studio, rocm), we have observed that their high CU utilization can delay other engines and cause TDRs. One such scenario is a compute workload preventing KIQ from dequeueing a GFX queue. When this happens, MES will see that the UNMAP_QUEUES packet has not completed in time (current timeout is 2 seconds), and it will pause scheduling and trigger a TDR. The current unmap sequence is this:

MES submits UNMAP_QUEUES packet to KIQ to unmap GFX queue
MES submits WRITE_DATA packet to KIQ with some fence
MES polls on fence memory to determine KIQ has consumed UNMAP_QUEUES packet
MES polls GFX queue's active bit to determine if queue has been unmapped successfully
If either steps 3 or 4 time out, then unmap has failed. This can happen because the compute queue is preventing the KIQ from dequeueing the GFX queue. However, we know that in some scenarios, if the timeout is ignored, then eventually everything would complete. This leads us to the following unmap sequence which utilizes the unmap all compute packet:

MES submits UNMAP_QUEUES packet to KIQ to unmap GFX queue
MES submits WRITE_DATA packet to KIQ with some fence
MES polls on fence memory to determine KIQ has consumed UNMAP_QUEUES packet
MES quickly polls the GFX queue's active bit to determine if queue has been unmapped successfully
If step 4 times out, unmap all compute
MES polls the GFX queue's active bit again
In this sequence, MES assumes that most dequeues will finish very quickly. At step 4, MES will implement a "short" wait on the active bit which should not timeout most of the time. If it does timeout, then MES will follow up by preempting the compute engine, which should free the KIQ so it can complete the GFX dequeue. If this preemption fails, then it's treated as an unmap timeout and TDR will occur. If preemption is successful, then MES will wait on the GFX queue's active bit again to determine if unmap is successful.

Compute Submission Impacting GFX QoS
Another issue caused by the high CU utilization from compute is that it simply prevents the currently mapped GFX queues from making forward progress. Our current workaround is to "time slice" the GFX and compute engines such that they are not allowed to run at the same time. The problem with the current implementation is that it has to be enabled by the driver when it detects particular problematic apps. This is not ideal for a couple of reasons: it's impossible to know every single problematic app, and it does not work for Linux environments.

The solution is to dynamically enable this time slicing when MES detects both compute and GFX queues are connected for some amount of time. This relies on the connect/idle interrupt in MES to keep track of which pipes are currently connected. When GFX and compute have both been connected for some amount of time, MES will use the unmap all compute packet to preempt the compute engine and allow GFX work to run on its own. It will also block compute queue scheduling. After a set duration, MES will unblock compute. The following state diagram outlines the flow: