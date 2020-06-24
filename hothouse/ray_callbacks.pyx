# cython: language=c++

from pyembree.callback_handler cimport \
    RayCollisionCallback, CALLBACK_TERMINATE, CALLBACK_CONTINUE
from pyembree.rtcore_ray cimport RTCRay

cdef class RayCollisionPrinter(RayCollisionCallback):
    cdef int callback(self, RTCRay ray):
        print("Hi!", ray.geomID)
        return CALLBACK_TERMINATE
