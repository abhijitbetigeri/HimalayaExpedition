import pathlib
S = pathlib.Path("/private/tmp/claude-501/-Users-abhijitbetigeri/9914382a-56e6-4234-b742-af0562bacf7e/scratchpad")
rock = (S / "rock.b64").read_text()
ice = (S / "ice.b64").read_text()

HTML = """<title>The Friction Gap</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&display=swap">
<style>
:root{
  --ink:#0d1721; --ground:#eef2f5; --surface:#ffffff; --muted:#5a6b7a;
  --line:#cfd9e0; --accent:#0f7fa6; --accent-soft:#e2f0f6;
  --fall:#b45309; --hold:#0f766e; --shadow:0 1px 2px rgba(13,23,33,.06),0 8px 24px rgba(13,23,33,.06);
  --serif:"IBM Plex Serif",Georgia,serif;
  --sans:"IBM Plex Sans",system-ui,-apple-system,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ink:#e6edf3; --ground:#0b141c; --surface:#121e28; --muted:#8ea3b4;
  --line:#25353f; --accent:#4bb8dd; --accent-soft:#132b36;
  --fall:#e09044; --hold:#4db6a4; --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.35);
}}
:root[data-theme="dark"]{
  --ink:#e6edf3; --ground:#0b141c; --surface:#121e28; --muted:#8ea3b4;
  --line:#25353f; --accent:#4bb8dd; --accent-soft:#132b36;
  --fall:#e09044; --hold:#4db6a4; --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);font-family:var(--sans);
  font-size:17px;line-height:1.65;-webkit-font-smoothing:antialiased}
.wrap{max-width:74ch;margin:0 auto;padding:clamp(2.5rem,6vw,5rem) clamp(1.1rem,4vw,2rem) 6rem;
  display:flex;flex-direction:column;gap:3.25rem}
.prose{display:flex;flex-direction:column;gap:1.15rem;max-width:65ch}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(2.4rem,6vw,3.6rem);
  line-height:1.04;letter-spacing:-.02em;margin:0;text-wrap:balance}
h2{font-family:var(--serif);font-weight:600;font-size:clamp(1.45rem,3.4vw,1.9rem);
  line-height:1.2;margin:0;text-wrap:balance;letter-spacing:-.01em}
h3{font-family:var(--sans);font-weight:600;font-size:1.02rem;margin:0;letter-spacing:-.005em}
p{margin:0}
.eyebrow{font-family:var(--mono);font-size:.7rem;letter-spacing:.16em;text-transform:uppercase;
  color:var(--accent);font-weight:500}
.lede{font-size:1.2rem;line-height:1.55;color:var(--muted);max-width:60ch}
.rule{height:1px;background:var(--line);border:0;margin:0}
strong{font-weight:600}
code,.mono{font-family:var(--mono);font-size:.88em}
a{color:var(--accent)}
section{display:flex;flex-direction:column;gap:1.5rem}

/* ---- the claim ---- */
.claim{background:var(--surface);border:1px solid var(--line);border-radius:2px;
  padding:1.6rem 1.75rem;box-shadow:var(--shadow);display:flex;flex-direction:column;gap:.9rem}
.claim .num{font-family:var(--mono);font-size:clamp(1.6rem,4.5vw,2.2rem);color:var(--accent);
  font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.claim p{color:var(--muted);font-size:.98rem}

/* ---- video comparison ---- */
.films{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1.1rem}
figure{margin:0;background:var(--surface);border:1px solid var(--line);border-radius:2px;
  overflow:hidden;box-shadow:var(--shadow);display:flex;flex-direction:column}
figure video{width:100%;display:block;background:#dfe6ea}
figcaption{padding:.85rem 1rem 1rem;display:flex;flex-direction:column;gap:.3rem;
  border-top:1px solid var(--line)}
.verdict{font-family:var(--mono);font-size:.78rem;letter-spacing:.04em;font-weight:500}
.verdict.hold{color:var(--hold)}
.verdict.fall{color:var(--fall)}
figcaption .mu{font-family:var(--mono);font-size:.82rem;color:var(--muted);
  font-variant-numeric:tabular-nums}

/* ---- tables ---- */
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:2px;background:var(--surface);
  box-shadow:var(--shadow)}
table{border-collapse:collapse;width:100%;font-size:.9rem;min-width:34rem}
th,td{text-align:left;padding:.6rem .85rem;border-bottom:1px solid var(--line);vertical-align:top}
thead th{font-family:var(--mono);font-size:.68rem;letter-spacing:.11em;text-transform:uppercase;
  color:var(--muted);font-weight:500;background:var(--accent-soft)}
tbody tr:last-child td{border-bottom:0}
td.n{font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
.tag{font-family:var(--mono);font-size:.7rem;letter-spacing:.05em;padding:.12rem .4rem;
  border-radius:2px;white-space:nowrap}
.tag.ok{background:var(--accent-soft);color:var(--hold)}
.tag.no{background:var(--accent-soft);color:var(--fall)}

/* ---- hypothesis ladder (a real sequence) ---- */
.ladder{display:flex;flex-direction:column;gap:0;border-left:2px solid var(--line);
  padding-left:0;margin-left:.4rem}
.step{display:grid;grid-template-columns:2.6rem 1fr;gap:.2rem 1rem;padding:.95rem 0 .95rem 1.15rem;
  border-bottom:1px solid var(--line);position:relative}
.step:last-child{border-bottom:0}
.step::before{content:"";position:absolute;left:-5px;top:1.35rem;width:8px;height:8px;
  border-radius:50%;background:var(--fall);border:2px solid var(--ground)}
.step.alive::before{background:var(--accent)}
.step .idx{font-family:var(--mono);font-size:.78rem;color:var(--muted);
  font-variant-numeric:tabular-nums;padding-top:.15rem}
.step .body{display:flex;flex-direction:column;gap:.3rem}
.step .h{font-weight:600;font-size:.97rem}
.step .why{font-size:.9rem;color:var(--muted)}
.step .why .mono{color:var(--ink)}

blockquote{margin:0;padding:.9rem 1.1rem;background:var(--accent-soft);
  border-left:2px solid var(--accent);font-size:.95rem;border-radius:0 2px 2px 0}
pre{margin:0;padding:1rem 1.1rem;background:var(--surface);border:1px solid var(--line);
  border-radius:2px;overflow-x:auto;font-family:var(--mono);font-size:.82rem;line-height:1.6;
  box-shadow:var(--shadow)}
.foot{font-size:.86rem;color:var(--muted);display:flex;flex-direction:column;gap:.5rem}
@media (prefers-reduced-motion:no-preference){
  video{transition:none}
}
</style>

<div class="wrap">

  <header class="prose">
    <span class="eyebrow">Himalaya Robotics Hack &middot; Track 1</span>
    <h1>Nobody taught the humanoid about ice.</h1>
    <p class="lede">Both major humanoid locomotion benchmarks train the Unitree G1 on
      ground roughly an order of magnitude grippier than ice &mdash; and one of them
      doesn&rsquo;t vary the friction at all.</p>
  </header>

  <section>
    <div class="claim">
      <span class="eyebrow">The number</span>
      <div class="num">&mu; = 0.8 &nbsp;&rarr;&nbsp; 0.6</div>
      <p>Isaac Lab&rsquo;s humanoid locomotion task sets static and dynamic friction as a
        <strong>point value, not a range</strong>. Minimum equals maximum, so the G1 sees exactly
        one surface for the whole of training. Real ice is &mu; = 0.05&ndash;0.15.</p>
    </div>
    <pre># isaaclab_tasks/.../locomotion/velocity/velocity_env_cfg.py

physics_material = EventTerm(
    func=mdp.randomize_rigid_body_material,
    params={"static_friction_range":  (0.8, 0.8),   # min == max
            "dynamic_friction_range": (0.6, 0.6)})</pre>
    <p class="prose">Neither G1 config overrides it. In the same source tree, the
      <em>quadruped</em> Spot gets <code>(0.3, 1.0)</code> and <code>(0.3, 0.8)</code>.
      The four-legged robot gets friction randomisation; the two-legged one doesn&rsquo;t.
      MuJoCo Playground is better but not by much &mdash; <code>U(0.4, 1.0)</code>, a floor
      still eight times above ice.</p>
  </section>

  <hr class="rule">

  <section>
    <h2>What that costs, on film</h2>
    <p class="prose">The same trained policy, the same seed, the same flat ground. The only
      difference between these two clips is the friction coefficient under the feet.</p>
    <blockquote>These are a single seed, and this baseline is not a reliable walker &mdash;
      only three of the first eight seeds reach 300 steps even on rock. Read the pair as an
      illustration of the failure mode, not as a measurement of it.</blockquote>
    <div class="films">
      <figure>
        <video src="data:video/mp4;base64,__ROCK__" controls muted loop playsinline preload="metadata"></video>
        <figcaption>
          <span class="verdict hold">HOLDS &mdash; 301 / 301 steps</span>
          <span class="mu">&mu; = 0.80 &middot; dry rock</span>
        </figcaption>
      </figure>
      <figure>
        <video src="data:video/mp4;base64,__ICE__" controls muted loop playsinline preload="metadata"></video>
        <figcaption>
          <span class="verdict fall">FALLS &mdash; step 178</span>
          <span class="mu">&mu; = 0.06 &middot; bare ice</span>
        </figcaption>
      </figure>
    </div>
    <p class="prose">The legs splay wider with each stride until the robot does the splits.
      It isn&rsquo;t a subtle degradation &mdash; the gait has no concept of a surface that
      slides.</p>
  </section>

  <hr class="rule">

  <section>
    <h2>Widening the range fixes it &mdash; in one simulator</h2>
    <p class="prose">Retraining the G1 with friction sampled across &mu; &isin; [0.05, 1.0]
      costs nothing on normal ground and covers a twenty-fold range.</p>
    <div class="scroll">
      <table>
        <thead><tr><th>Isaac Lab</th><th>Baseline &mu;=0.8 fixed</th><th>Ice-trained &mu; 0.05&ndash;1.0</th></tr></thead>
        <tbody>
          <tr><td>Success rate</td><td class="n">1.000</td><td class="n">1.000</td></tr>
          <tr><td>Survives full episode</td><td class="n">98.0%</td><td class="n">97.8%</td></tr>
          <tr><td>Velocity tracking</td><td class="n">0.913</td><td class="n">0.912</td></tr>
          <tr><td>Velocity error (m/s)</td><td class="n">0.133</td><td class="n">0.134</td></tr>
        </tbody>
      </table>
    </div>
    <blockquote>Each column is measured on its own training distribution, so this is not yet
      a like-for-like A/B. It shows the ice policy pays no tax for the wider range &mdash;
      not that it beats the baseline on ice.</blockquote>
  </section>

  <hr class="rule">

  <section>
    <h2>The same experiment fails in MuJoCo</h2>
    <p class="prose">Identical friction range, comparable step budget, opposite result.
      Five explanations were tested and four are dead. This is the honest part.</p>
    <div class="ladder">
      <div class="step"><span class="idx">01</span><div class="body">
        <span class="h">The eval harness is broken</span>
        <span class="why">Refuted. The baseline reaches <span class="mono">301/300</span>
          steps through the same code path, while the ice arm never passes 47 on any
          seed. A broken harness could not produce that gap.</span></div></div>
      <div class="step"><span class="idx">02</span><div class="body">
        <span class="h">Log-uniform sampling starves it of easy ground</span>
        <span class="why">Refuted. Switching to uniform still failed &mdash;
          <span class="mono">35.1 &plusmn; 6.4</span> of 501 on stock flat terrain.</span></div></div>
      <div class="step"><span class="idx">03</span><div class="body">
        <span class="h">The slip penalty punishes the surface, not the gait</span>
        <span class="why">Refuted. Relaxing <span class="mono">feet_slip</span> from
          &minus;0.25 to &minus;0.05 changed nothing: <span class="mono">35.9</span> vs
          <span class="mono">35.6</span>.</span></div></div>
      <div class="step"><span class="idx">04</span><div class="body">
        <span class="h">Too many novel conditions at once</span>
        <span class="why">Refuted. Three of the four factors break walking
          <em>individually</em>; it is not an interaction effect.</span></div></div>
      <div class="step"><span class="idx">05</span><div class="body">
        <span class="h">Simply undertrained</span>
        <span class="why">Refuted. Six times the budget &mdash; 25M to 150M steps &mdash;
          moved survival from <span class="mono">7.1%</span> to
          <span class="mono">10.7%</span>. Isaac reaches 98% at the same 150M.</span></div></div>
      <div class="step"><span class="idx">06</span><div class="body">
        <span class="h">A gait-phase reward forces a fixed cadence</span>
        <span class="why">Refuted. Playground rewards <span class="mono">feet_phase</span>
          at +1.0 and Isaac has no phase term, so this looked like the structural
          difference. Zeroing it changed nothing: <span class="mono">37.6 &plusmn; 6.8</span>
          against <span class="mono">35.6 &plusmn; 7.2</span>.</span></div></div>
      <div class="step alive"><span class="idx">&mdash;</span><div class="body">
        <span class="h">Still unexplained</span>
        <span class="why">Six explanations tested, six dead. Isaac reaches 98% on this
          exact experiment and MuJoCo Playground reaches 10.7% at a matched step budget,
          and we cannot yet say why.</span></div></div>
    </div>
  </section>

  <hr class="rule">

  <section>
    <h2>Which factor actually breaks walking</h2>
    <p class="prose">One variable changed per run, 16 rollouts each, scored on stock flat
      ground so the control is identical throughout.</p>
    <div class="scroll">
      <table>
        <thead><tr><th>Run</th><th>Changed</th><th>Survival</th><th></th></tr></thead>
        <tbody>
          <tr><td class="mono">A2</td><td>Compliant ground (soft snow)</td><td class="n">390.5 &plusmn; 137.5</td><td><span class="tag ok">78% &mdash; harmless</span></td></tr>
          <tr><td class="mono">A3</td><td>Wind</td><td class="n">65.8 &plusmn; 15.9</td><td><span class="tag no">13%</span></td></tr>
          <tr><td class="mono">A1</td><td>Friction range alone</td><td class="n">35.6 &plusmn; 7.2</td><td><span class="tag no">7%</span></td></tr>
          <tr><td class="mono">A4</td><td>Mid-episode rock&rarr;ice patches</td><td class="n">11.6 &plusmn; 2.6</td><td><span class="tag no">2% &mdash; worst</span></td></tr>
        </tbody>
      </table>
    </div>
    <p class="prose">Soft, deformable ground is <em>fine</em>. Sudden transitions between
      surfaces are the most destructive thing you can do to a humanoid gait &mdash; worse
      than ice itself.</p>
  </section>

  <hr class="rule">

  <section>
    <h2>Where it stands</h2>
    <p class="prose">The finding is verified in both source trees and does not depend on any
      policy training well. One simulator now walks across a twenty-fold friction range at no
      cost to normal-ground performance. The other refuses to, for a reason that is down to
      one live hypothesis.</p>
    <div class="foot">
      <p><strong>Not claimed:</strong> that the ice policy beats the baseline on ice &mdash;
        that needs the cross-evaluation, which isn&rsquo;t finished. That the MuJoCo clips
        generalise &mdash; they are one seed from a policy that walks on a minority of them.
        No hardware was used. Fall-step counts are not comparable across physics backends and
        are quoted from one backend throughout.</p>
      <p><span class="mono">Unitree G1 &middot; Isaac Lab &amp; MuJoCo Playground &middot;
        trained on Hugging Face Jobs</span></p>
    </div>
  </section>

</div>
"""

HTML = HTML.replace("__ROCK__", rock).replace("__ICE__", ice)
out = pathlib.Path("/Users/abhijitbetigeri/projects/himalaya-hack/friction_gap.html")
out.write_text(HTML)
print("wrote", out, f"{len(HTML)/1024/1024:.2f} MB")
