# 🎯 Easy Supabase Setup with Environment Variables

This guide shows you how to set up Supabase using simple environment variables instead of connection strings.

## ✅ What You Need from Supabase

Go to **Supabase Dashboard** → **Settings** → **API** and get:

1. **Project URL** (SUPABASE_URL)
   - Format: `https://lgbpmgaacqawvfavtdzu.supabase.co`
   - Found in: **Project URL** section

2. **Database Password** (SUPABASE_DB_PASSWORD)
   - This is the password you set when creating the project
   - Your password: `NAgyxHiwctATJwro`
   - Can be reset in: **Settings** → **Database** → **Database password**

3. **Anon Key** (SUPABASE_ANON_KEY) - Optional
   - Found in: **Project API keys** → **anon** `public` key
   - Used for Supabase client library (future use)

## 📝 Step 1: Get Your Values from Supabase

1. **Go to Supabase Dashboard**
   - Visit: https://supabase.com
   - Sign in and open your project

2. **Get Project URL**
   - Go to **Settings** → **API**
   - Copy **Project URL**: `https://lgbpmgaacqawvfavtdzu.supabase.co`

3. **Get Database Password**
   - Go to **Settings** → **Database**
   - Your password: `NAgyxHiwctATJwro`
   - Or reset it if needed

4. **Get Anon Key** (Optional)
   - Go to **Settings** → **API**
   - Copy **anon** `public` key

## 🔧 Step 2: Set Environment Variables in Vercel

### Option A: Using Vercel Dashboard (Recommended)

1. **Go to Vercel Dashboard**
   - Visit: https://vercel.com
   - Open your project

2. **Add Environment Variables**
   - Go to **Settings** → **Environment Variables**
   - Click **Add New** for each variable:

   **Variable 1: SUPABASE_URL**
   - **Key**: `SUPABASE_URL`
   - **Value**: `https://lgbpmgaacqawvfavtdzu.supabase.co`
   - **Environment**: ✅ Production, ✅ Preview, ✅ Development

   **Variable 2: SUPABASE_DB_PASSWORD**
   - **Key**: `SUPABASE_DB_PASSWORD`
   - **Value**: `NAgyxHiwctATJwro`
   - **Environment**: ✅ Production, ✅ Preview, ✅ Development

   **Variable 3: SUPABASE_ANON_KEY** (Optional)
   - **Key**: `SUPABASE_ANON_KEY`
   - **Value**: (Your anon key from Supabase)
   - **Environment**: ✅ Production, ✅ Preview, ✅ Development

3. **Click Save** for each variable

4. **Redeploy** your project

### Option B: Using .env File (Local Development)

1. **Create `.env.local` file** in project root:
   ```bash
   SUPABASE_URL=https://lgbpmgaacqawvfavtdzu.supabase.co
   SUPABASE_DB_PASSWORD=NAgyxHiwctATJwro
   SUPABASE_ANON_KEY=your-anon-key-here
   JWT_SECRET_KEY=your-secret-key
   ```

2. **Load in your app** (already configured ✅)

## 🎯 How It Works

The code automatically:
1. ✅ Reads `SUPABASE_URL` and `SUPABASE_DB_PASSWORD`
2. ✅ Extracts project reference from URL
3. ✅ Builds PostgreSQL connection string automatically
4. ✅ Connects to your Supabase database

**No need to manually create connection strings!**

## 📋 Your Values (Copy & Paste)

```bash
# Add these to Vercel Environment Variables:

SUPABASE_URL=https://lgbpmgaacqawvfavtdzu.supabase.co
SUPABASE_DB_PASSWORD=NAgyxHiwctATJwro
```

## ✅ Verification

After setting variables and deploying:

1. **Check Debug Endpoint**
   - Visit: `https://your-app.vercel.app/api/debug/auth`
   - Should show: `"database_type": "PostgreSQL"`
   - Should show: `"database_accessible": true`

2. **Test Login**
   - Email: `patient@demo.com`
   - Password: `patient123`

## 🔄 Alternative: Direct Connection String

If you prefer the old method, you can still use:
```bash
DATABASE_URL=postgresql://postgres:NAgyxHiwctATJwro@db.lgbpmgaacqawvfavtdzu.supabase.co:5432/postgres
```

But the new method is **easier** and **more secure**! ✅

## 🆘 Troubleshooting

### Database Not Connecting
- Verify `SUPABASE_URL` format: `https://[PROJECT_REF].supabase.co`
- Check `SUPABASE_DB_PASSWORD` is correct
- Ensure variables are set for all environments

### Wrong Project Reference
- The code extracts project ref from URL automatically
- Make sure URL format is correct

### Still Using SQLite
- Check Vercel logs to see if env vars are being read
- Verify variables are set for the correct environment
- Redeploy after adding variables

---

**Much easier than connection strings!** 🎉

