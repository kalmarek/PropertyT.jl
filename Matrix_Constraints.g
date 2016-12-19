Symmetrise := function(elts)
  return Unique(Concatenation(elts, List(elts, Inverse)));
end;

MYAllProducts := function(elts1, elts2)
  local products, elt;
  products := [];
  for elt in elts1 do
    products := Concatenation(products, elt*elts2);
  od;
  return products;
end;

Products := function(elts, n)
  local products, i;
  if n<=0 then
      return [ ];
  elif n = 1 then
      return elts;
  else
    products := elts;
    for i in [2..n] do
      products := MYAllProducts(elts, products);
    od;
    return products;
  fi;
end;

IsSupportedOn := function(basis, elt)
    local elt_supp, x;
    elt_supp := Support(elt);
    for x in elt_supp do
        if not x in basis then
            return false;
        fi;
    od;
    return true;
end;

Laplacian := function(G, generating_set)
  local QG, emb, result, S, g, elt;
  QG := GroupRing(Rationals, G);;
  emb := Embedding(G,QG);;

  S := generating_set;

  result := Length(S)*One(QG);
  for g in S do
      result := result - g^emb;
  od;
  return result;
end;

Vectorise := function(elt, basis)
  local result, l, i, g, coeff, axis;
  Assert(0, IsSupportedOn(basis, elt),
    "AssertionError: Element of interest is not supported on the basis!");
  result := List(0*[1..Length(basis)]);

  l := CoefficientsAndMagmaElements(elt);
  for i in [1..Length(l)/2] do
    g := l[2*i-1];
    coeff := l[2*i];
    axis := Position(basis, g);
    result[axis] := result[axis] + coeff;
  od;
  return result;
end;

Constraints := function(basis)
  local result, i, j, pos;
  result := [];
  for i in [1..Length(basis)] do
    Add(result,[]);
  od;
  for i in [1..Length(basis)] do
     for j in [1..Length(basis)] do
       pos := Position(basis, Inverse(basis[i])*basis[j]);
       if not pos = fail then
          Add(result[pos], [i,j]);
       fi;
     od;
  od;
  return result;
end;

SDPGenerateAll := function(G, S, basis, name)
  local QG, emb, delta, delta_sq, delta_vec, delta_sq_vec, product_constr;
  QG := GroupRing(Rationals, G);;
  emb := Embedding(G,QG);;

  delta := Laplacian(G, S);;
  delta_sq := delta^2;;
  if not IsSupportedOn(basis, delta_sq) then
    #   Print("delta_sq is not supported on basis\n");
      return fail;
  else
      PrintTo(Concatenation("./basis.", name), basis);
      Print("Written basis to ", Concatenation("./basis.", name), "\n");
      delta_vec := Vectorise(delta, basis);;
      PrintTo(Concatenation("./delta.", name), delta_vec);
      Print("Written delta to ", Concatenation("./delta.", name), "\n");
      delta_sq_vec := Vectorise(delta_sq, basis);;
      PrintTo(Concatenation("./delta_sq.", name), delta_sq_vec);
      Print("Written delta_sq to ", Concatenation("./delta_sq.", name), "\n");

      product_constr := Constraints(basis);;
      PrintTo(Concatenation("./constraints.", name), product_constr);
      Print("Written Matrix Constraints to ", Concatenation("./Constraints.", name), "\n");
      return "Done!";
  fi;

end;;
