from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
my_cart = comm.Create_cart([2, 2], False, False)
print(f"proc {rank}: coords = {my_cart.Get_coords(rank)}...")
prev, next = my_cart.Shift(1, 1)
print(f"proc {rank}: prev = {prev}, next = {next}...")
comm.Sendrecv(prev, next, 0, next, prev, 0, None)
print(f"proc {rank} sent {prev} and received {next}...")
