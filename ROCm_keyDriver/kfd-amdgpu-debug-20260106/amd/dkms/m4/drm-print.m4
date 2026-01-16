dnl #
dnl # commit v6.14-rc4-376-gd05386a3fdf3
dnl # drm/print: require struct drm_device for drm_err() and friends
dnl #
AC_DEFUN([AC_AMDGPU_DRM_TO_DEV], [
    AC_KERNEL_DO_BACKGROUND([
        AC_KERNEL_TRY_COMPILE([
            #include <drm/drm_print.h>
        ], [
            __drm_to_dev(NULL);
        ], [
            AC_DEFINE(HAVE_DRM_TO_DEV, 1, [
                __drm_to_dev is available])
        ])
    ])
])
