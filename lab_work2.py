from numba import cuda
print(f"Available GPU : {cuda.is_available()}")
print(f'list of GPU : {cuda.detect()}')
dev=cuda.select_device(0)
print('Device name: ', dev.name)
print(f'number of multiprocessor: {dev.MULTIPROCESSOR_COUNT}')
print(f'Total memory size: {cuda.current_context().get_memory_info().total}')