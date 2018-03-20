⭐⭐ Slitherin’. Implement and solve a multiplayer clone of the classic Snake game (see slither.io for inspiration) as a Gym environment.


Gym 링크
1. https://github.com/openai/gym


환경:

큰 field 에 여러마리 뱀이 있다. 

뱀은 랜덤하게 나타나는 과일을 먹고 자란다.

다른 뱀이나 벽에 충돌하면 죽으며, 모든 뱀이 죽으면 게임이 끝난다. 

두마리의 뱀으로 시작한다. 

에이전트 : 

self-play를 사용하여 내가 선택한 강화학습 알고리즘으로 위의 환경을 풀어야한다. 

self-play의 instability를 풀기위해 여러가지로 접근하여 실험해야 한다. 

예를들어, 현재 폴리
Environment: have a reasonably large field with multiple snakes; snakes grow when eating randomly-appearing fruit; a snake dies when colliding with another snake, itself, or the wall; and the game ends when all snakes die. Start with two snakes, and scale from there.
Agent: solve the environment using self-play with an RL algorithm of your choice. You’ll need to experiment with various approaches to overcome self-play instability (which resembles the instability people see with GANs). For example, try training your current policy against a distribution of past policies. Which approach works best?
Inspect the learned behavior: does the agent learn to competently pursue food and avoid other snakes? Does the agent learn to attack, trap, or gang up against the competing snakes? Tweet us videos of the learned policies!
