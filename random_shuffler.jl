using LatinSquares

# v::Vector{String}, length(v) == 24
function latin_with_fixed_first_column(v::Vector{String})
    # build the standard cyclic Latin square of order 24:
    #   latin(n)[i,j] == ((i + j - 2) % n) + 1
    n = lastindex(v)
    S = latin(n)        # 24×24 Array{Int,2} with 1..24 in each row & column :contentReference[oaicite:0]{index=0}

    # now replace each integer k by v[k]:
    M = Array{String}(undef, n, n)
    @inbounds for i in 1:n, j in 1:n
        M[i, j] = v[S[i, j]]
    end

    return M
end

