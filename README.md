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

## Run examples
```bash
python latentdem_inference.py \
  -x 30 \
  -y 70 \
  -z 0 \
  -i '../data/images/01_shoe_image.txt' \
  -o '../data/results/latentdem.png' \
  -p '../data/images/01_shoe_pose.txt' \
  --sample_from 'middle' \
  --Estep_scheduling 'linear' \
  --Mstep_scheduling 'linear' \
  --lr 10 \
  --lr_decay \
  --skip_Mstep \
```

- `x, y, z`: Processing camera transform
- `i`: Input images, in a txt format
- `o`: Output image (novel view) location
- `p`: Initial pose based on first image
- `sample_from`: Sampling method for phi2, phi3, .... One of **start/middle/prev**
- `Estep_scheduling`: Scheduling method (gamma) for E-step. One of **linear/fixed/curve**
- `Mstep_scheduling`: Scheduling method (lambda) for M-step. One of **linear/fixed**
- `lr`: Learning rate value for M-step
- `lr_decay`: If true, lr decreases by timestep
- `skip_Mstep`: If true, phis are not optimized and fixed by initial position

---

If you meet `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`, follow
- `apt-get update -y`
- `apt-get install -y libgl1-mesa-glx`

  
