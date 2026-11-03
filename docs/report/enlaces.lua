-- In the PDF a relative link leads nowhere, so it points at the file in the repository.
local repository = "https://github.com/Joakamakaka1/machine-learning-project/blob/master/"

function Link(link)
  if link.target:match("^%a[%w+.-]*:") or link.target:match("^#") then
    return nil
  end
  local path = link.target:gsub("^%.%./", ""):gsub("^%./", "")
  if not link.target:match("^%.%./") then
    path = "docs/" .. path
  end
  link.target = repository .. path
  return link
end
