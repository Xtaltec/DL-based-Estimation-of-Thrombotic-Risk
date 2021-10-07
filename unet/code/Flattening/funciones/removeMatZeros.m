function B = removeMatZeros(A)
    B = [];
    for i = 1: size(A, 1)
        r = A(i,:);
        r(r==0) = []; % remove zeros
          % handle expansion
          ncolR = size(r, 2);
          ncolB = size(B, 2);
          diffcol = ncolR - ncolB;
          if (diffcol > 0) % previous rows need more cols
              for j = ncolB+1:ncolR
                  B(:,j) = NaN;
              end
          elseif (diffcol < 0) % this row needs more cols
              r = [r, NaN(1, abs(diffcol))];
          end
          B(i,:) = r;
      end
  end