/* The name of the Metal device, for the backend report. Objective-C, because
 * Metal's device API is. Nothing else in VxVector uses it. */
#import <Metal/Metal.h>

#include <stddef.h>
#include <string.h>

int vxvector_gpu_device_name(char *name, size_t size) {
    if (!name || size == 0)
        return 0;
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            name[0] = '\0';
            return 0;
        }
        strlcpy(name, device.name.UTF8String, size);
        return 1;
    }
}
