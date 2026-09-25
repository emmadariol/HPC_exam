"""Small SVG plotting helper with the same look as the figures made by analyze.py."""
import math

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
GREY = "#555"
F = 'font-family="sans-serif"'


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Chart:
    def __init__(self, title, xlabel, ylabel, xlabels, ymin, ymax, yticks=None, ylog=False,
                 yfmt="{:.2g}", pad=False, legend="tl", width=900, height=520):
        self.w, self.h = width, height
        self.l, self.t, self.r, self.b = 82, 42, 35, 75
        self.pw, self.ph = width - self.l - self.r, height - self.t - self.b
        n = len(xlabels)
        self.x0, self.x1 = (-0.5, n - 0.5) if pad else (0, max(1, n - 1))  # x positions are category indices
        if yticks is None and not ylog:  # round the y range to a nice step
            raw = (ymax - ymin) / 5
            e = 10 ** math.floor(math.log10(raw))
            step = min((m * e for m in (1, 2, 2.5, 5, 10)), key=lambda st: abs(math.log(st / raw)))
            ymin, ymax = math.floor(ymin / step + 1e-9) * step, math.ceil(ymax / step - 1e-9) * step
            yticks = [ymin + step * i for i in range(round((ymax - ymin) / step) + 1)]
        self.ylog, self.y0, self.y1 = ylog, ymin, ymax
        self.legend_pos, self.nleg = legend, 0
        self.p = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{width/2}" y="28" text-anchor="middle" {F} font-size="20">{esc(title)}</text>',
            f'<line x1="{self.l}" y1="{height-self.b}" x2="{width-self.r}" y2="{height-self.b}" stroke="black"/>',
            f'<line x1="{self.l}" y1="{self.t}" x2="{self.l}" y2="{height-self.b}" stroke="black"/>',
            f'<text x="{width/2}" y="{height-18}" text-anchor="middle" {F} font-size="14">{esc(xlabel)}</text>',
            f'<text transform="translate(24,{height/2}) rotate(-90)" text-anchor="middle" {F} font-size="14">{esc(ylabel)}</text>',
        ]
        for i, lab in enumerate(xlabels):
            self.p.append(f'<text x="{self.X(i):.1f}" y="{height-self.b+28}" text-anchor="middle" {F} font-size="12">{esc(lab)}</text>')
        if yticks is None:
            if ylog:
                yticks = [10.0 ** e for e in range(round(math.log10(ymin)), round(math.log10(ymax)) + 1)]
                yfmt = "1e{:.0f}"
        for yv in yticks:
            y = self.Y(yv)
            lab = yfmt.format(math.log10(yv)) if ylog and yfmt == "1e{:.0f}" else yfmt.format(yv)
            self.p.append(f'<line x1="{self.l-5}" y1="{y:.1f}" x2="{self.l}" y2="{y:.1f}" stroke="black"/>')
            self.p.append(f'<text x="{self.l-10}" y="{y+4:.1f}" text-anchor="end" {F} font-size="12">{esc(lab)}</text>')

    def X(self, i):
        return self.l + self.pw * (i - self.x0) / (self.x1 - self.x0)

    def Y(self, v):
        if self.ylog:
            f = (math.log10(v) - math.log10(self.y0)) / (math.log10(self.y1) - math.log10(self.y0))
        else:
            f = (v - self.y0) / (self.y1 - self.y0)
        return self.t + self.ph * (1.0 - f)

    def line(self, xs, ys, color, dash=None, r=4, label=None):
        pts = " ".join(f"{self.X(x):.1f},{self.Y(y):.1f}" for x, y in zip(xs, ys))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.p.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"{d}/>')
        if r:
            self.points(xs, ys, color, r)
        if label:
            self.legend(label, color)

    def points(self, xs, ys, color, r=4):
        for x, y in zip(xs, ys):
            self.p.append(f'<circle cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{r}" fill="{color}"/>')

    def hline(self, y, color=GREY, dash="6,4"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.p.append(f'<line x1="{self.l}" y1="{self.Y(y):.1f}" x2="{self.w-self.r}" y2="{self.Y(y):.1f}" stroke="{color}" stroke-width="2"{d}/>')

    def seg(self, x1, y1, x2, y2, color="black", width=2):
        self.p.append(f'<line x1="{self.X(x1):.1f}" y1="{self.Y(y1):.1f}" x2="{self.X(x2):.1f}" y2="{self.Y(y2):.1f}" stroke="{color}" stroke-width="{width}"/>')

    def errorbar(self, x, y, err, color, ecolor=GREY):
        cap = 6
        xp, lo, hi = self.X(x), self.Y(y - err), self.Y(y + err)
        self.p.append(f'<line x1="{xp:.1f}" y1="{lo:.1f}" x2="{xp:.1f}" y2="{hi:.1f}" stroke="{ecolor}" stroke-width="2"/>')
        for yy in (lo, hi):
            self.p.append(f'<line x1="{xp-cap:.1f}" y1="{yy:.1f}" x2="{xp+cap:.1f}" y2="{yy:.1f}" stroke="{ecolor}" stroke-width="2"/>')
        self.points([x], [y], color, 5)

    def text(self, x, y, s, dx=0, dy=0, color="#333", anchor="start", size=12):
        self.p.append(f'<text x="{self.X(x)+dx:.1f}" y="{self.Y(y)+dy:.1f}" text-anchor="{anchor}" {F} font-size="{size}" fill="{color}">{esc(s)}</text>')

    def note(self, px, py, s, color="#333", size=13):
        self.p.append(f'<text x="{px}" y="{py}" {F} font-size="{size}" fill="{color}">{esc(s)}</text>')

    def legend(self, label, color):
        k = self.nleg
        self.nleg += 1
        right = self.legend_pos[1] == "r"
        x = self.w - self.r - 10 if right else self.l + 10
        y = self.t + 20 + 20 * k if self.legend_pos[0] == "t" else self.h - self.b - 15 - 20 * (3 - k)
        anchor = "end" if right else "start"
        self.p.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" {F} font-size="13" fill="{color}">{esc(label)}</text>')

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.p + ["</svg>"]))
        print(f"wrote {path}")
