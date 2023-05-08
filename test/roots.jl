using PropertyT.Roots
@testset "Roots" begin
    @test Roots.Root{3,Int}([1, 2, 3]) isa Roots.AbstractRoot{}
    @test Roots.Root([1, 2, 3]) isa Roots.AbstractRoot{3,Int}
    # io
    r = Roots.Root{3,Int}([1, 2, 3])
    @test contains(sprint(show, MIME"text/plain"(), r), "of length √14\n")
    r = Roots.Root{3,Int}([1, 2, 2])
    @test contains(sprint(show, MIME"text/plain"(), r), "of length 3\n")
end
