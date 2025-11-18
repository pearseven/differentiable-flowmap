# An Adjoint Method for Differentiable Fluid Simulation on Flow Maps

This repository contains the official implementation of SIGGRAPH Asia Conference Paper  
**“An Adjoint Method for Differentiable Fluid Simulation on Flow Maps”**  
By Zhiqi Li, Jinjin He, Barnabás Börcsök, Taiyuan Zhang, Duowen Chen, Tao Du, Ming C. Lin, Greg Turk & Bo Zhu.  
Paper available at [arXiv: 2511.01259](https://arxiv.org/abs/2511.01259) (Nov 2025).

> 🔗 **Project Page / Demo:** [https://pearseven.github.io/DiffFMProject/](https://pearseven.github.io/DiffFMProject/)  
> 📄 [**Paper PDF / DOI** ](https://arxiv.org/abs/2511.01259)  
> 🎥 [**Video Demo** ](https://www.youtube.com/watch?v=C1RrUa53uxU)

---
## Build Instructions

This implementation is written entirely in Python 3.9 using Taichi for high-performance simulation.

### Dependencies

Install all required packages:

pip install taichi numpy pillow matplotlib scipy


or using a requirements.txt:
```
taichi>=1.5.0
numpy
pillow
matplotlib
scipy
```

Then install with:
```
pip install -r requirements.txt
```
## Usage
For 2D G->R smoke shape transition
```
cd 2D/shape
python optimize_shape_G_R.py
```
For 2D vortex optimization
```
cd 2D/vortex
python optimize_vortex.py
```
For 3D G->R smoke shape transition
```
cd 3D
optimize_shape_G_R.py
```
## Citation

If you find this repository helpful, please cite:

```bibtex
@article{li2025adjoint,
  title   = {An Adjoint Method for Differentiable Fluid Simulation on Flow Maps},
  author  = {Li, Zhiqi and He, Jinjin and B\"orcs\"ok, Barnab\'as and Zhang, Taiyuan and Chen, Duowen and Du, Tao and Lin, Ming C. and Turk, Greg and Zhu, Bo},
  journal = {SIGGRAPH Asia Conference Papers},
  year    = {2025},
  note    = {arXiv:2511.01259},
  url     = {https://arxiv.org/abs/2511.01259}
}
