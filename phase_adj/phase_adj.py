from sys import argv
import h5py
import numpy as np
from scipy import *
from scipy.optimize import least_squares
from matplotlib.pylab import *
from PySide import QtCore, QtGui
import pycuda.driver as cuda
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
from timeit import default_timer as timer

dataset = 'garrido'
if len(argv) > 1:
    dataset = argv[1]

cuda.Device(0).make_context()
cuda.Context.set_cache_config(cuda.func_cache.PREFER_L1)

f = h5py.File('fids.h5','r')
dset = f[dataset]
a = dset[...]
t = arange(0,len(a[0,:]),1)

def Norm_spectrum(s):
	mini=min(s)
	s=s-mini
	maxi=max(s)
	s=s/maxi
	return s

def ACMEentropy(x, s, ref_ph1):
	stepsize = 1
	func_type = 1
	
	L = len(s)
	phc0 = x[0]
	phc1 = x[1]
	a_num = -(array(range(1,L + 1)) - ref_ph1)/float(L)	

	s0 = s * exp(1j*(pi/180)*(phc0 + phc1 * a_num))
	s0 = s0.real

	ds1 = abs((s0[3:L]-s0[1:L-2])/(stepsize*2))
	p1 = ds1 / sum(ds1)

	p1[np.where(p1 == 0)[0]] = 1
	h1 = -p1 * log(p1)
	H1 = sum(h1)
	
	Pfun = 0.0
	as2 = s0 - abs(s0)
	sumas = sum(as2)

	if sumas < 0:
		Pfun = Pfun + sum(pow(as2, 2))/float(4*L*L)
	P = 1000 * Pfun
	return [H1 + P, 0]

    
fid_complexo=a[0,t]+a[1,t]*1j
FFT = fft(fid_complexo)
FFT = np.fft.fftshift(FFT)
#FFT.real=Norm_spectrum(FFT.real)
#FFT.imag=Norm_spectrum(FFT.imag)
freqs = array(range(0, fid_complexo.size))    

size = len(FFT.real)
#print('Working on dataset %s of size %d' % (dataset, size))

subplot(221)
plot(freqs, FFT.real)
subplot(222)
plot(freqs, FFT.imag)

## compute gpu phase correction

kernel = """
#define N %d
#define DIV_N  (1.0 / N)
#define DIV_H1 (250.0 * DIV_N * DIV_N)
#define RAD_1  (3.1415926535898 / 180)

__device__ __constant__ float device_real[N];
__device__ __constant__ float device_imag[N];
__device__ float entropia[1];
__device__ int lock = 0;

__device__ float fatomicMin(float *addr, float value)
{
    float old = *addr, assumed;
    if(old <= value) return old;
    do {
        assumed = old;
        old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));
    } while(old!=assumed);
    return old;
}

__global__ void phase_adj(int *phc0, int *phc1, int *pivot)
{
    const int pivot_local = blockIdx.y;
    const int phc0_local  = blockIdx.x - 180;
    const int phc1_local = threadIdx.x - 180;

    float ds1[N];
    float sum_as  = 0;
    float sum_as2 = 0;
    float sum_ds1 = 0;
    float H1 = 0;
    int i;

    // Complex Rotation
    for(i = 0; i < N; i++)
    {
        float theta = RAD_1 * (phc0_local + phc1_local * DIV_N * (pivot_local-i-1));
        float s = device_real[i] * __cosf(theta) - device_imag[i] * __sinf(theta);

        float as = s -fabs(s);
        sum_as  += as;
        sum_as2 += as * as;

        ds1[i] = s;
        if (i > 1) {
            ds1[i-2] = fabs((s - ds1[i-2]) * 0.5);
            sum_ds1 += ds1[i-2];
        }
    }
    // Calculation of Entropy
    for(i = 0; i < N-2; i++)
    {
        if (ds1[i] != 0.0) {
            ds1[i] /= sum_ds1;
            H1 += -ds1[i] * log(ds1[i]);
        }
    }
    if (sum_as < 0)
        H1 += sum_as2 * DIV_H1;

    if (H1 < entropia[0]) {
        fatomicMin(entropia, H1);
        int needlock = 1;
        while(needlock)
        {
            if(atomicCAS(&lock, 0, 1) == 0)
            {
                *phc0 = phc0_local;
                *phc1 = phc1_local;
                *pivot = pivot_local;
                atomicExch(&lock, 0);
                needlock = 0;
            }
        }
    }
}
""" % (size)

mod = SourceModule(kernel, options=["--ptxas-options=-v"])

start = timer()

phase_adj = mod.get_function("phase_adj")

entropia_gpu = mod.get_global('entropia')[0]
device_real = mod.get_global('device_real')[0]
device_imag = mod.get_global('device_imag')[0]

cuda.memcpy_htod(device_real,  FFT.real.astype(np.float32))
cuda.memcpy_htod(device_imag,  FFT.imag.astype(np.float32))
entropia_cpu = array([2147483647])
cuda.memcpy_htod(entropia_gpu, entropia_cpu.astype(np.float32))
phc0_cpu = array([0])
phc0_gpu = cuda.to_device(phc0_cpu.astype(np.int32))
phc1_cpu = array([0])
phc1_gpu = cuda.to_device(phc1_cpu.astype(np.int32))
pivot_cpu = array([0])
pivot_gpu = cuda.to_device(pivot_cpu.astype(np.int32))

phase_adj(
    phc0_gpu,
    phc1_gpu,
    pivot_gpu,
    block=(360,1,1), #fase 0
    grid=(360,size)) #fase 1, pivot

entropia_cpu = cuda.from_device_like(entropia_gpu, entropia_cpu.astype(np.float32))
phc0_cpu = cuda.from_device_like(phc0_gpu, phc0_cpu.astype(np.int32))
phc1_cpu = cuda.from_device_like(phc1_gpu, phc1_cpu.astype(np.int32))
pivot_cpu = cuda.from_device_like(pivot_gpu, pivot_cpu.astype(np.int32))
gpu_time = timer() - start
gpu_entropy = ACMEentropy((phc0_cpu, phc1_cpu), FFT, pivot_cpu)[0]

def adjust(s, phc0, phc1, ref_ph1):
    L = len(s)
    s0 = s * exp(1j*(pi/180)*(phc0 + phc1 * (-(array(range(1,L + 1)) - ref_ph1)/float(L))))
    return s0

s = adjust(FFT, phc0_cpu, phc1_cpu, pivot_cpu)

subplot(2,2,3)
plot(freqs, s.real)
subplot(2,2,4)
plot(freqs, s.imag)

## validate with cpu least squares

start = timer()
pivots = [x for x in range(size)]
pivots.append(pivot_cpu)
np.random.shuffle(pivots)
sample = pivots[:min(size,99)]
sample.append(pivot_cpu)
#sample = range(1,size+1)
results = []
for pivot in sample:
    xopt = least_squares(ACMEentropy, [ 0.0, 0.0 ], bounds=(-180, 180), args=(FFT, pivot))
    x = xopt.x
    h = ACMEentropy(x,FFT,pivot)[0]
    results.append((h,pivot,x))
x = min(results)
cpu_time = (timer()-start) / len(sample) * size # estimated time to run all pivots
cpu_entropy = x[0]

# dataset(size), cpu_entropy, gpu_entropy, cpu_time, gpu_time, speedup")
print("%s(%d)\t%f\t%f\t%f\t%f\t%f" % (dataset, size, cpu_entropy, gpu_entropy, cpu_time, gpu_time, cpu_time/gpu_time))

show()
cuda.Context.pop()
