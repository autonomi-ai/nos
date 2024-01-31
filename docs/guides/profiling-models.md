## ⏱️ Profiling Models

`nos profile` supports model benchmarking across a number of axes including iterations per second,
memory footprint and utilization on CPU/GPU.
Currently, the nos profiler itself runs natively in the execution environment (i.e. outside
the NOS server), so you'll need to install both the `server` and `test` dependencies alongside
your existing NOS installation 

```bash
pip install torch-nos[server,test]
```

If you have the time, you can construct a profiling catalog on your machine in its entirety with:
```bash
nos profile rebuild-catalog
```

You can also profile specific models with 
```bash
nos profile method -m openai/clip-vit-large-patch14
```

Or an entire method type with
```bash
nos profile method -m encode_image
```
to benchmark e.g. all image embedding models.

Dump the nos profiling catalog with `nos profile list`

![NOS Profile List](../assets/nos_profile_list.png)