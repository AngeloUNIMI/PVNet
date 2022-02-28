from util.slugify import slugify

def selectInverter(folderInv, sapm_inverters):

    for inverter_count, inverter_name in enumerate(sapm_inverters):

        # sanitize
        inverter_name_slug = slugify(inverter_name)

        # if slugified version is equal to folder name, select it
        if inverter_name_slug == folderInv:
            inverter = sapm_inverters[inverter_name]
            return inverter
