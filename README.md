# SFL-RPi

![License](https://img.shields.io/github/license/TIANGeng708/SFL-RPi) ![Python](https://img.shields.io/badge/python-3.7%2B-blue)

## 🔥 项目简介
**SFL-RPi** 是一个基于 **树莓派 (Raspberry Pi)** 构建的 **分割联邦学习 (SplitFed Learning, SFL)** 训练平台，支持对 **收敛速率**、**通信效率** 和 **能耗情况** 进行测试。

### ✨ 主要特性
- **真实分布式环境**：树莓派作为客户端，个人电脑作为服务器，模拟真实的边缘计算环境。
- **WiFi 无线通信**：采用 **HTTP 协议** 保障数据完整、高效传输。
- **灵活扩展**：支持多客户端动态加入，适用于不同实验需求。
- **低成本、易部署**：基于树莓派搭建，适合研究和教学场景。
- **PyTorch 实现**，兼容主流深度学习框架。

## 🚀 快速开始
### 1️⃣ 环境依赖
在服务器（PC）和客户端（树莓派）上安装以下依赖：
```bash
pip install torch torchvision requests numpy matplotlib
```

### 2️⃣ 克隆仓库
```bash
git clone https://github.com/TIANGeng708/SFL-RPi.git
cd SFL-RPi
```

### 3️⃣ 启动服务器
在 PC 上运行服务器端：
```bash
python server.py
```

### 4️⃣ 启动客户端
在树莓派上运行客户端：
```bash
python clientxx.py  <服务器IP>
```

## ⚙️ 主要功能
| 功能 | 描述 |
|------|------|
| **支持的训练方法** | SplitFed Learning (SFL) |
| **通信协议** | WiFi + HTTP |
| **硬件支持** | Raspberry Pi (客户端) + PC (服务器) |
| **可测指标** | 收敛速率、通信效率、能耗 |

## 📌 贡献指南
欢迎提交 **Issue** 和 **Pull Request** 来优化本项目！

## 📎 相关链接
- 🔗 GitHub 仓库：[SFL-RPi](https://github.com/TIANGeng708/SFL-RPi)

