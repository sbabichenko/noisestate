# Notes

Standalone pages for things that want a figure and an equation more than they want a paragraph.
They are plain HTML: open one in a browser, or publish it as an artifact.  Markdown in `docs/`
stays the reference; a note here is an argument with its numbers attached.

| note | what it holds |
|---|---|
| [best-response-jacobian.html](best-response-jacobian.html) | why `res.stability()`'s spectral radius is `rho(D^-1 O)`, read off the first-order condition, and why the second-order check cannot see what it sees |

`_template.html` is that page with the content taken out: the tokens, the type scale and the
components, with the constraints written into the comment at the top.  Copy it, change the
`<title>`, and write.

## The constraints, because they fail silently

* **Fonts** load only from `fonts.googleapis.com`.  A face from anywhere else falls back with no
  error, so always declare a real fallback stack.
* **KaTeX and MathJax do not work here.**  Their stylesheets and font files are both off the
  artifact CSP's allowlist, so the maths would render as unstyled markup.  The template sets it by
  hand: `.m` for an italic symbol, `.frac` for a stacked derivative.  That covers a first-order
  condition and its Hessians, which is most of what these notes need.
* **Both themes, three states.**  The viewer's "system" setting stamps nothing on the root element,
  so every colour is a token declared in the bare `:root` and redefined in *both* the
  `prefers-color-scheme` block and the `[data-theme="dark"]` block.  A colour that exists only
  inside one of those renders one theme's text on the other theme's ground.
* **`body` paints its own background** from a token.  A transparent body borrows the host's ground.

## Numbers

Every figure in a note is measured, and the note says where from.  The Jacobian note's numbers come
from `extras/tools/ch3_long_window_branch.py` and a direct assembly of the 48x48 Jacobian described
in its own footer; `docs/limits.md` carries the same table in prose.
