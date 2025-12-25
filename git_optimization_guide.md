# 大型Git仓库推送问题解决方案

## 问题分析
从错误信息可以看出，推送失败的主要原因：
1. 数据包过大(2.03 GiB)
2. 连接重置(RPC failed; curl 55 Recv failure)
3. 远程端意外挂断

## 解决方法

### 1. 增加Git缓冲区大小
```bash
# 临时增加缓冲区大小
git config http.postBuffer 524288000  # 500MB
git config http.maxRequestBuffer 1048576000  # 1GB

# 或者全局设置
git config --global http.postBuffer 524288000
git config --global http.maxRequestBuffer 1048576000
```

### 2. 清理历史提交中的大型文件
使用git-filter-repo工具移除已提交的大文件：

```bash
# 安装git-filter-repo
pip install git-filter-repo

# 查找大文件
git filter-repo --analyze

# 移除特定目录（如果它们已被提交）
git filter-repo --force --invert-paths --path data/dataset_image/ --path wandb/ --path pictures/
```

### 3. 分批推送策略
```bash
# 先推送部分历史
git push origin --depth=100 master

# 然后逐步增加深度
git push origin --depth=500 master
git push origin master
```

### 4. 使用Git LFS管理大文件
```bash
# 安装Git LFS
git lfs install

# 追踪图像文件
git lfs track "*.jpg"
git lfs track "*.png"

# 提交.gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

### 5. 强制推送（谨慎使用）
```bash
# 仅在必要时使用
git push --force origin master
```

### 6. 其他优化建议
- 检查并关闭不必要的代理
- 使用SSH而不是HTTPS协议
- 确认网络稳定性

## 注意事项
1. 使用git-filter-repo会重写历史，确保在执行前备份仓库
2. Git LFS需要服务器支持
3. 操作前确保没有未提交的更改
