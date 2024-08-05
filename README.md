## Settings

1. Make folder `weights` under `zero123`, and add `zero123-xl.ckpt`

```
.
├── zero123
│   └── weights
│       └── zero123-xl.ckpt
└── ...
```

2. Install conda environment, same with [Zero123](https://github.com/cvlab-columbia/zero123)

3. Test with
```bash
python latentdem_inference.py -x 0 -y 90 -z 0 -i images.txt -o result.png
```

---

If you meet `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`, follow
- `apt-get update -y`
- `apt-get install -y libgl1-mesa-glx`

  
