from jinja2 import Environment, PackageLoader

prompt_loader = Environment(
        loader=PackageLoader("lrg", package_path="prompting/templates")
    )