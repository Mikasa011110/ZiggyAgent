## 简介
本项目实现了一个面部表情识别（FER）与面试场景分析系统：

- **FER模型**：在FER2013和DAF-DB数据集上训练，基于预训练ResNet50模型，并对部分层进行了微调（fine-tuning）。
- **LLM功能**：受硬件和时间限制，语言模型部分直接调用DeepSeek API。
- **技术栈**：前后端基于Flask搭建，前端展示界面，后端提供识别API，部署在本地服务器（localhost）。
- **项目时间与背景**：2025年7月完成于南京大学苏州校区暑期学校。

## 🚀 How to Use
打开START.bat，等待一会儿，用浏览器打开链接即可

# 注意！！！您可能无法直接使用
- 1. 用于面部表情识别的模型：ResNet50.pt，在训练时保存的是全部参数，只能在特定版本的pytorch上运行
  2. 我没有续费DeepSeek API

## ⚖️ License
MIT License

![海报](./Poster.png)
