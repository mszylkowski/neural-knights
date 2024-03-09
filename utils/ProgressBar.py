from tqdm import tqdm


class ProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{"colour": "green", "unit_scale": True, **kwargs})

    def set(self, val):
        super().update(val - self.n)


format_number = tqdm.format_sizeof
