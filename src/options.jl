###
### Swarm coefficients
###

mutable struct StaticCoeffs{T}
    w::T
    c1::T
    c2::T

    function StaticCoeffs(w, c1, c2)
        any(<(0), (w, c1, c2)) && throw(ArgumentError("Coefficients can't be negative."))
        T = promote_type(typeof.((w, c1, c2))...)
        return new{T}(w, c1, c2)
    end
end

StaticCoeffs(; w=1.0, c1=2.0, c2=2.0) = StaticCoeffs(w, c1, c2)

function Base.setproperty!(c::StaticCoeffs, sym::Symbol, val)
    if sym in (:w, :c1, :c2) && val < 0
        throw(ArgumentError("Coefficient $sym can't be negative."))
    end
    return setfield!(c, sym, val)
end

coefficients(sc::StaticCoeffs) = (sc.w, sc.c1, sc.c2)

###
### TODO: Swarm topology
###