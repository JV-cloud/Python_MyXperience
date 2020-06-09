import camelcase

c = camelcase.CamelCase()

txt = "neste exemplo todas as primeiras letras de cada palavra ficarão maiúsculas"

print(c.hump(txt))