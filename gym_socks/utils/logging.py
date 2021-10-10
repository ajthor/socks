from tqdm.auto import tqdm

_progress_fmt = "|{bar:30}| {elapsed_ms}"


class ms_tqdm(tqdm):
    @property
    def format_dict(self):
        d = super(ms_tqdm, self).format_dict

        min, s = divmod(d["elapsed"], 60)
        h, m = divmod(min, 60)

        if h:
            elapsed_ms = "{0:d}:{1:02d}:{2:06.3f}".format(int(h), int(m), s)
        else:
            elapsed_ms = "{0:02d}:{1:06.3f}".format(int(m), s)

        d.update(elapsed_ms=elapsed_ms)
        return d
