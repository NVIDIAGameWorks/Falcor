struct Ball
{
    float2 pos;
    float2 vel;
};

RWStructuredBuffer<Ball> g_balls;
uniform uint g_ball_count;
uniform float g_dt;

[numthreads(128, 1, 1)]
void main(uint3 thread_id: SV_DispatchThreadID)
{
    uint idx = thread_id.x;
    if (idx >= g_ball_count)
        return;

    Ball ball = g_balls[idx];

    ball.pos += ball.vel * g_dt;
    if (ball.pos.x < -1.f || ball.pos.x > 1.f)
    {
        ball.pos.x = sign(ball.pos.x);
        ball.vel.x *= -1.f;
    }
    if (ball.pos.y < -1.f || ball.pos.y > 1.f)
    {
        ball.pos.y = sign(ball.pos.y);
        ball.vel.y *= -1.f;
    }

    g_balls[idx] = ball;
}
