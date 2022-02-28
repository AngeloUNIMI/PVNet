from util.slugify import slugify

def selectModule(folderModule, sandia_modules):

    for module_count, module_name in enumerate(sandia_modules):

        # sanitize
        module_name_slug = slugify(module_name)

        # if slugified version is equal to folder name, select it
        if module_name_slug == folderModule:
            module = sandia_modules[module_name]
            return module
