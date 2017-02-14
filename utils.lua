utils = {}

-- create a reds vs. blues dataset (two gaussian clouds with same isotropic variance)
function utils.make_data(N, stdev)
    local reds = torch.Tensor(N/2,2)
    local blues = torch.Tensor(N/2,2)
    reds[{{}, 1}]:normal(5, stdev)
    reds[{{}, 2}]:normal(10, stdev)
    blues[{{}, 1}]:normal(10, stdev)
    blues[{{}, 2}]:normal(5, stdev)
    return {reds, blues}
end


-- utility function, returns a string representation of a slice of a vector v from f to t (inclusive)
-- works for tables and tensors
function utils.vec2strp(v, f, t)
    local r = '['
    for i=f,t do
        r = r .. string.format('%g', v[i])
        if i < t then r = r .. ', ' end
    end
    return r .. ']'
end


-- returns a string representing the entire vector (works for tables and tensors)
function utils.vec2str(v)
    local n = 0
    if type(v) == 'table' then n = #v else n = v:size(1) end
    return utils.vec2strp(v, 1, n)
end


-- prints the output of the rule f with parameters ps on input xs
function utils.print_rule_output(xs, ps, f)
    local res = f(xs, ps)
    print('Perceptron with weights ' .. utils.vec2strp(ps,1,2) .. ' and bias ' .. ps[3] .. ' maps ' .. utils.vec2str(xs) .. ' to ' .. res)
end

-- maps the scalar function f over all values of tensor xs (indexed by dimension 1)
-- and returns the result
function utils.map(f, xs)
    local ys = torch.Tensor(xs:size())
    local n = xs:size(1)
    for i=1,n do
        ys[i] = f(xs[i])
    end
    return ys
end

-- returns min(max(v, low), high)
function utils.cull(v,low,high)
    if v < low then v = low end
    if v > high then v = high end
    return v
end

-- given the parameters of the perceptron in a table or tensor {w1, w2, b}
-- this function returns two endpoints inside the [0,20] x [0,20] square for
-- plotting
function utils.perceptron_separator(ps)
    local w1, w2, b = unpack(ps)
    -- x1 * w1 + x2 * w2 + b = 0
    local xx1 = torch.Tensor({-20, 20})
    local xx2 = -(xx1 * w1 + b) / w2
    xx2[1] = utils.cull(xx2[1],-20, 20)
    xx2[2] = utils.cull(xx2[2],-20, 20)
    xx1 = -(xx2 * w2 + b) / w1
    return xx1, xx2
end

function utils.enlarge(img, factor)
    local sz = img:size()
    local ofs = 0

    if sz:size(1) == 3 then
        local m, n = sz[2]*factor, sz[3]*factor
        img2 = torch.DoubleTensor(sz[1],m,n)
        for k=1,sz[1] do
            for i=1,sz[2]*factor do
                for j=1,sz[3]*factor do
                    img2[k][i][j] = img[k][(i+factor-1)/factor][(j+factor-1)/factor]
                end
            end
        end
        return img2
    elseif sz:size(1) == 2 then
        local m, n = sz[1]*factor, sz[2]*factor
        img2 = torch.DoubleTensor(m, n)
        for i=1,sz[1]*factor do
            for j=1,sz[2]*factor do
                img2[i][j] = img[(i+factor-1)/factor][(j+factor-1)/factor]
            end
        end
        return img2
    else
        error('Invalid tensor dimensions, must be 2 or 3.')
    end
end

