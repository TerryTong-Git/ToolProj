\* Graph_Coloring *\
Maximize
OBJ: __dummy
Subject To
_C1: Colors_(1,_'A') + Colors_(1,_'B') + Colors_(1,_'C') = 1
_C10: Colors_(1,_'A') + Colors_(5,_'A') <= 1
_C11: Colors_(1,_'B') + Colors_(5,_'B') <= 1
_C12: Colors_(1,_'C') + Colors_(5,_'C') <= 1
_C13: Colors_(2,_'A') + Colors_(3,_'A') <= 1
_C14: Colors_(2,_'B') + Colors_(3,_'B') <= 1
_C15: Colors_(2,_'C') + Colors_(3,_'C') <= 1
_C16: Colors_(2,_'A') + Colors_(4,_'A') <= 1
_C17: Colors_(2,_'B') + Colors_(4,_'B') <= 1
_C18: Colors_(2,_'C') + Colors_(4,_'C') <= 1
_C19: Colors_(2,_'A') + Colors_(6,_'A') <= 1
_C2: Colors_(2,_'A') + Colors_(2,_'B') + Colors_(2,_'C') = 1
_C20: Colors_(2,_'B') + Colors_(6,_'B') <= 1
_C21: Colors_(2,_'C') + Colors_(6,_'C') <= 1
_C22: Colors_(3,_'A') + Colors_(5,_'A') <= 1
_C23: Colors_(3,_'B') + Colors_(5,_'B') <= 1
_C24: Colors_(3,_'C') + Colors_(5,_'C') <= 1
_C25: Colors_(4,_'A') + Colors_(6,_'A') <= 1
_C26: Colors_(4,_'B') + Colors_(6,_'B') <= 1
_C27: Colors_(4,_'C') + Colors_(6,_'C') <= 1
_C28: Colors_(5,_'A') + Colors_(6,_'A') <= 1
_C29: Colors_(5,_'B') + Colors_(6,_'B') <= 1
_C3: Colors_(3,_'A') + Colors_(3,_'B') + Colors_(3,_'C') = 1
_C30: Colors_(5,_'C') + Colors_(6,_'C') <= 1
_C4: Colors_(4,_'A') + Colors_(4,_'B') + Colors_(4,_'C') = 1
_C5: Colors_(5,_'A') + Colors_(5,_'B') + Colors_(5,_'C') = 1
_C6: Colors_(6,_'A') + Colors_(6,_'B') + Colors_(6,_'C') = 1
_C7: Colors_(1,_'A') + Colors_(3,_'A') <= 1
_C8: Colors_(1,_'B') + Colors_(3,_'B') <= 1
_C9: Colors_(1,_'C') + Colors_(3,_'C') <= 1
Bounds
 __dummy = 0
Binaries
Colors_(1,_'A')
Colors_(1,_'B')
Colors_(1,_'C')
Colors_(2,_'A')
Colors_(2,_'B')
Colors_(2,_'C')
Colors_(3,_'A')
Colors_(3,_'B')
Colors_(3,_'C')
Colors_(4,_'A')
Colors_(4,_'B')
Colors_(4,_'C')
Colors_(5,_'A')
Colors_(5,_'B')
Colors_(5,_'C')
Colors_(6,_'A')
Colors_(6,_'B')
Colors_(6,_'C')
End
