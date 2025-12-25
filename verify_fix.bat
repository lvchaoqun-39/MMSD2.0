@echo off

REM Git推送问题验证脚本
REM 此脚本将执行一系列步骤来验证和修复Git推送问题

echo === Git推送问题验证开始 ===

REM 1. 清理Git缓存以应用新的.gitignore规则
echo 步骤1: 清理Git缓存，应用新的.gitignore规则...
git rm -r --cached .
git add .
git status
echo .gitignore更新已应用

REM 2. 增加缓冲区大小
echo 步骤2: 增加Git缓冲区大小...
git config http.postBuffer 524288000
git config http.maxRequestBuffer 1048576000
echo 缓冲区大小已设置

REM 3. 验证仓库大小
echo 步骤3: 显示当前仓库状态...
git status
echo Git对象存储信息:
git count-objects -v

REM 4. 提供推送建议
echo === 验证完成 ===
echo 请执行以下命令进行推送:
echo 1. 先尝试普通推送:
git push origin master

echo 2. 如果仍然失败，尝试分批推送:
git push origin --depth=100 master

REM 5. 显示完整优化指南位置
echo 请参考 git_optimization_guide.md 文件获取更多解决方案

pause
