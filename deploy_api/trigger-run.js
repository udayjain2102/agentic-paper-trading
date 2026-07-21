// Vercel serverless function backing the dashboard's "run new research run"
// button. Copied into site/api/ at deploy time (see the workflow's deploy
// step) since the deployed site/ dir is regenerated fresh on every run.
//
// Requires one Vercel project env var, set once by hand in the Vercel
// dashboard (Project -> Settings -> Environment Variables):
//   GITHUB_DISPATCH_TOKEN - a GitHub PAT (fine-grained: Actions "read and
//                           write" on this repo; classic: "repo" scope)
//
// The endpoint is unauthenticated: anyone with the URL can trigger a run.
// That only kicks off a GitHub Actions research run (no money, no data), so
// the blast radius is CI spam. Re-add a shared-secret check if that matters.
export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'POST only' });
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
