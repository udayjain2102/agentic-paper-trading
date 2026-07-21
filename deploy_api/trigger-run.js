// Vercel serverless function backing the dashboard's "run new research run"
// button. Copied into site/api/ at deploy time (see the workflow's deploy
// step) since the deployed site/ dir is regenerated fresh on every run.
//
// Requires two Vercel project env vars, set once by hand in the Vercel
// dashboard (Project -> Settings -> Environment Variables):
//   RUN_TRIGGER_SECRET   - passphrase the button prompts for; pick anything
//   GITHUB_DISPATCH_TOKEN - a GitHub PAT (fine-grained: Actions "read and
//                           write" on this repo; classic: "repo" scope)
export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'POST only' });
    return;
  }
  const { secret } = req.body || {};
  if (!secret || secret !== process.env.RUN_TRIGGER_SECRET) {
    res.status(401).json({ error: 'unauthorized' });
    return;
  }
  const token = process.env.GITHUB_DISPATCH_TOKEN;
  if (!token) {
    res.status(500).json({ error: 'GITHUB_DISPATCH_TOKEN not configured' });
    return;
  }

  const ghRes = await fetch(
    'https://api.github.com/repos/udayjain2102/papertrading-model-/actions/workflows/research-run.yml/dispatches',
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: 'application/vnd.github+json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ ref: 'main' }),
    }
  );

  if (ghRes.status === 204) {
    res.status(200).json({ ok: true });
    return;
  }
  const detail = await ghRes.text();
  res.status(502).json({ error: 'dispatch failed', detail });
}
