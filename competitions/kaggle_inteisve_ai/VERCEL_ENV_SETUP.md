# 🔐 Vercel Environment Variables Setup

## Quick Setup (Copy & Paste)

Go to **Vercel Dashboard** → Your Project → **Settings** → **Environment Variables**

### Required Variables:

#### 1. DATABASE_URL (REQUIRED)
```
postgresql://postgres:NAgyxHiwctATJwro@db.lgbpmgaacqawvfavtdzu.supabase.co:5432/postgres
```
- **Key**: `DATABASE_URL`
- **Value**: `postgresql://postgres:NAgyxHiwctATJwro@db.lgbpmgaacqawvfavtdzu.supabase.co:5432/postgres`
- **Environment**: ✅ Production, ✅ Preview, ✅ Development

#### 2. JWT_SECRET_KEY (REQUIRED)
```
your-secret-key-change-in-production
```
- **Key**: `JWT_SECRET_KEY`
- **Value**: (Generate a secure random string)
- **Environment**: ✅ Production, ✅ Preview, ✅ Development

#### 3. GEMINI_API_KEY (Optional)
```
your-gemini-api-key-here
```
- **Key**: `GEMINI_API_KEY`
- **Value**: (Your Gemini API key if using LLM features)
- **Environment**: ✅ Production, ✅ Preview, ✅ Development

## Steps:

1. **Add DATABASE_URL**:
   - Click **Add New**
   - Key: `DATABASE_URL`
   - Value: `postgresql://postgres:NAgyxHiwctATJwro@db.lgbpmgaacqawvfavtdzu.supabase.co:5432/postgres`
   - Select all environments
   - Save

2. **Add JWT_SECRET_KEY** (if not already set):
   - Click **Add New**
   - Key: `JWT_SECRET_KEY`
   - Value: Generate with: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`
   - Select all environments
   - Save

3. **Redeploy**:
   - Go to **Deployments** tab
   - Click **⋯** on latest deployment
   - Click **Redeploy**

## ✅ Verify Setup

After redeploy, check:
```
https://your-app.vercel.app/api/debug/auth
```

Should show:
- `"database_type": "PostgreSQL"`
- `"database_accessible": true`
- `"user_count": 3` (after demo users initialize)

## 🎯 Test Login

- Email: `patient@demo.com`
- Password: `patient123`

---

**Once DATABASE_URL is set, login will work!** 🚀

