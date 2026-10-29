-- Pandoc filters used by build.js when rendering posts.

-- Pandoc can't number equations, so it turns \eqref{label} into a link that
-- just shows "[label]". Hand \eqref back to MathJax instead, which numbers
-- equation/align environments and renders the reference as "(1)".
function Link(el)
  if el.attributes["reference-type"] == "eqref" then
    return pandoc.Math("InlineMath", "\\eqref{" .. el.attributes["reference"] .. "}")
  end
end

-- Pandoc reads every LaTeX list as "loose" (each item wrapped in <p>, with
-- extra spacing). Make lists whose items are single paragraphs tight.
local function tighten(list)
  for _, item in ipairs(list.content) do
    if #item == 1 and item[1].t == "Para" then
      item[1] = pandoc.Plain(item[1].content)
    end
  end
  return list
end
BulletList = tighten
OrderedList = tighten
