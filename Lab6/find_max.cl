__kernel void find_max(__global unsigned int *data)
{ 
    int id = get_global_id(0);
    data[id] = max(data[id * 2], data[id * 2 + 1]);
}