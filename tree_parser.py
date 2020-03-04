def find_end_of_subtree(sentence, pos):
  if sentence[pos] != "(":
  	return pos
  sum = 1
  while sum > 0:
  	pos += 1
  	if sentence[pos] == "(":
  	  sum += 1
  	elif sentence[pos] == ")":
  	  sum -= 1
  	if sum == 0:
  	  return pos
  return -1

def is_parsable2(sentence):
  ret = is_parsable(sentence)
  print(sentence, ret)
  return ret

def is_parsable3(sentence):
  if sentence[0] != "(":
  	return False
  if find_end_of_subtree(sentence, 0) != len(sentence) - 1:
  	return False
  if len(sentence) == 3:
  	return True

  pos = find_end_of_subtree(sentence, 2) + 1
  if not is_parsable2(sentence[2:pos]):
  	return False

  if sentence[1] == 'c' or sentence[1] == 'v' or sentence[1] == 'cart':
    return pos + 2 == len(sentence) 
  else:
   	return is_parsable2(sentence[pos:(len(sentence) - 1)])

def is_parsable(sentence):
  if len(sentence) == 1:
  	return True

  if sentence[0] != '(':
  	return False
  if find_end_of_subtree(sentence, 0) + 1 != len(sentence):
  	return False
  if len(sentence) == 3:
  	return True

  pos = 2
  while pos + 1 < len(sentence):
  	npos = find_end_of_subtree(sentence, pos) + 1
  	if npos == 0:
  	  return False
  	if not is_parsable(sentence[pos:npos]):
  	  return False
  	pos = npos

  return True

def split_into_subtrees(sentence, max_size):
  if (len(sentence) <= 1):
    return []

  if len(sentence) <= max_size:
  	return [sentence]

  ret = []
  pos = 2
  while pos + 1 < len(sentence):
  	npos = find_end_of_subtree(sentence, pos) + 1
  	ret.extend(split_into_subtrees(sentence[pos:npos], max_size))
  	pos = npos

  return ret

def get_small_subtrees(sentence, min_size, max_size):
  if len(sentence) < min_size:
    return []

  ret = []
  if (len(sentence) <= max_size):
    ret.append(sentence)

  pos = 2
  while pos + 1 < len(sentence):
    npos = find_end_of_subtree(sentence, pos) + 1
    ret.extend(get_small_subtrees(sentence[pos:npos], min_size, max_size))
    pos = npos

  return ret

def find_subtree(tree, subtree):
  for i in range (len(tree)):
    for j in range (len(subtree)):
      if (tree[i + j] != subtree[j]):
        break

      if (j == len(subtree ) - 1):
        return i

# tree = "(a (a (c (fun (bool) (fun (bool) (bool))) /\\) (a (a (c (fun (real) (fun (real) (bool))) real_lt) (a (c (fun (num) (real)) real_of_num) (a (c (fun (num) (num)) NUMERAL) (c (num) _0)))) (v (real) q))) (a (c (fun (fun (cart (real) N) (bool)) (bool)) !) (l (v (cart (real) N) x') (a (a (c (fun (bool) (fun (bool) (bool))) ==>) (a (a (c (fun (bool) (fun (bool) (bool))) /\\) (a (a (c (fun (cart (real) N) (fun (fun (cart (real) N) (bool)) (bool))) IN) (v (cart (real) N) x')) (v (fun (cart (real) N) (bool)) t))) (a (a (c (fun (real) (fun (real) (bool))) real_lt) (a (c (fun (prod (cart (real) N) (cart (real) N)) (real)) distance) (a (a (c (fun (cart (real) N) (fun (cart (real) N) (prod (cart (real) N) (cart (real) N)))) ,) (v (cart (real) N) y)) (v (cart (real) N) x')))) (v (real) q)))) (a (a (c (fun (bool) (fun (bool) (bool))) /\\) (a (c (fun (fun (cart (real) M) (bool)) (bool)) !) (l (v (cart (real) M) x) (a (a (c (fun (bool) (fun (bool) (bool))) ==>) (a (a (c (fun (bool) (fun (bool) (bool))) /\\) (a (a (c (fun (cart (real) M) (fun (fun (cart (real) M) (bool)) (bool))) IN) (v (cart (real) M) x)) (v (fun (cart (real) M) (bool)) s))) (a (a (c (fun (cart (real) N) (fun (cart (real) N) (bool))) =) (a (v (fun (cart (real) M) (cart (real) N)) f) (v (cart (real) M) x))) (v (cart (real) N) y)))) (a (c (fun (fun (cart (real) M) (bool)) (bool)) ?) (l (v (cart (real) M) x') (a (a (c (fun (bool) (fun (bool) (bool))) /\\) (a (a (c (fun (bool) (fun (bool) (bool))) /\\) (a (a (c (fun (cart (real) M) (fun (fun (cart (real) M) (bool)) (bool))) IN) (v (cart (real) M) x')) (v (fun (cart (real) M) (bool)) s))) (a (a (c (fun (cart (real) N) (fun (cart (real) N) (bool))) =) (a (v (fun (cart (real) M) (cart (real) N)) f) (v (cart (real) M) x'))) (v (cart (real) N) x')))) (a (a (c (fun (real) (fun (real) (bool))) real_lt) (a (c (fun (prod (cart (real) M) (cart (real) M)) (real)) distance) (a (a (c (fun (cart (real) M) (fun (cart (real) M) (prod (cart (real) M) (cart (real) M)))) ,) (v (cart (real) M) x)) (v (cart (real) M) x')))) (v (real) d))))))))) (a (c (fun (fun (cart (real) M) (bool)) (bool)) !) (l (v (cart (real) M) x) (a (a (c (fun (bool) (fun (bool) (bool))) ==>) (a (a (c (fun (bool) (fun (bool) (bool))) /\\) (a (a (c (fun (cart (real) M) (fun (fun (cart (real) M) (bool)) (bool))) IN) (v (cart (real) M) x)) (v (fun (cart (real) M) (bool)) s))) (a (a (c (fun (cart (real) N) (fun (cart (real) N) (bool))) =) (a (v (fun (cart (real) M) (cart (real) N)) f) (v (cart (real) M) x))) (v (cart (real) N) x')))) (a (c (fun (fun (cart (real) M) (bool)) (bool)) ?) (l (v (cart (real) M) x') (a (a (c (fun (bool) (fun (bool) (bool))) /\\) (a (a (c (fun (bool) (fun (bool) (bool))) /\\) (a (a (c (fun (cart (real) M) (fun (fun (cart (real) M) (bool)) (bool))) IN) (v (cart (real) M) x')) (v (fun (cart (real) M) (bool)) s))) (a (a (c (fun (cart (real) N) (fun (cart (real) N) (bool))) =) (a (v (fun (cart (real) M) (cart (real) N)) f) (v (cart (real) M) x'))) (v (cart (real) N) y)))) (a (a (c (fun (real) (fun (real) (bool))) real_lt) (a (c (fun (prod (cart (real) M) (cart (real) M)) (real)) distance) (a (a (c (fun (cart (real) M) (fun (cart (real) M) (prod (cart (real) M) (cart (real) M)))) ,) (v (cart (real) M) x)) (v (cart (real) M) x')))) (v (real) d)))))))))))))(a (a (c (fun (bool) (fun (bool) (bool))) =) (a (a (c (fun (real) (fun (real) (bool))) real_le) (a (c (fun (cart (real) (1)) (real)) drop) (v (cart (real) (1)) b))) (a (c (fun (cart (real) (1)) (real)) drop) (v (cart (real) (1)) d)))) (c (bool) T))"

# subs = split_into_subtrees(tree, 1000)
# subs = [x for x in subs if len(x) > 200]

# subs = get_small_subtrees(tree, 50, 100)

# for sub in subs:
#   a = find_subtree(tree, sub)
#   print (a)


# print(tree[97:])
# print()
# print(subs[0])
# for i in range (len(subs)):
#   print (subs[i])

# print(len(tree))

# nums = [1, 2, 3, 4]
# nues = nums[1:-1]
# print (3 * 1/4)

# li = ['3', 're' , 'lsdf']

# lu = list(li)
# print(lu)
