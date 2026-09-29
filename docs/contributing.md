# 贡献指南

1. Fork 仓库，新建特性分支；
2. 遵循零依赖内核约定（core 层不引入第三方依赖）；
3. 新增后端实现 `Backend` 接口并在 `__init__.py` 注册；
4. 提交前跑 `python -m terraforme doctor` 与 `python -m unittest discover -s tests -t .`；
5. 提交 PR。
