import taichi as ti

ti.init(arch=ti.cuda, device_memory_GB=8.0, debug = False,default_fp = ti.f64)
a = ti.field(float ,shape = (100,100))
print(a.shape)
b = a.to_numpy()
print(b.shape)
b=b[:50,:]
print(b.shape)
a.from_numpy(b)
