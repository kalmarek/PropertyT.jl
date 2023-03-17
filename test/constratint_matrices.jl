@testset "ConstraintMatrix" begin
    @test PropertyT.ConstraintMatrix{Float64}(
        [-1, 2, -1, 1, 4, 2, 6],
        3,
        2,
        π,
    ) isa AbstractMatrix

    cm = PropertyT.ConstraintMatrix{Float64}([-1, 2, -1, 1, 4, 2, 6], 3, 2, π)

    @test cm == Float64[
        -π π
        2π 0
        0 π
    ]

    @test collect(PropertyT.nzpairs(cm)) == [
        1 => 3.141592653589793
        2 => 3.141592653589793
        2 => 3.141592653589793
        4 => 3.141592653589793
        6 => 3.141592653589793
        1 => -3.141592653589793
        1 => -3.141592653589793
    ]

    @test PropertyT.ConstraintMatrix{Float64}([-9:-1; 1:9], 3, 3, 1.0) ==
          zeros(3, 3)
end
