
import taichi as ti

ti.init(arch=ti.gpu)

@ti.func
def mul(v1: ti.f32, v2: ti.f32) -> ti.f32:
    return v1 * v2

@ti.kernel
def vec_mul(v1: ti.template(), v2: ti.template(), target: ti.template()):
    for i in range(v1.shape[0]):
        target[i] = mul(v1[i], v2[i])


@ti.kernel
def initialize_fields(vec: ti.template()):
    for i in range(vec.shape[0]):
        vec[i] = 2.0

size: int = int(1e8)

v1 = ti.field(dtype=ti.f32, shape=size)
v2 = ti.field(dtype=ti.f32, shape=size)
v3 = ti.field(dtype=ti.f32, shape=size)

# all set to 0
initialize_fields(v1)
initialize_fields(v2)

vec_mul(v1, v2, v3)

print(v3)

