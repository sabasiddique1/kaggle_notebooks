# 🎯 Simple Environment Variables Setup

## ✅ What You Need

Just **2 environment variables** from Supabase:

1. **SUPABASE_URL** - Your project URL
2. **SUPABASE_DB_PASSWORD** - Your database password

## 📋 Your Values (Copy These)

```bash
SUPABASE_URL=https://lgbpmgaacqawvfavtdzu.supabase.co
SUPABASE_DB_PASSWORD=NAgyxHiwctATJwro
```

## 🚀 How to Add in Vercel

1. **Go to Vercel Dashboard** → Your Project
2. **Settings** → **Environment Variables**
3. **Add Variable 1:**
   - Key: `SUPABASE_URL`
   - Value: `https://lgbpmgaacqawvfavtdzu.supabase.co`
   - Environments: ✅ All
4. **Add Variable 2:**
   - Key: `SUPABASE_DB_PASSWORD`
   - Value: `NAgyxHiwctATJwro`
   - Environments: ✅ All
5. **Save** and **Redeploy**

## ✅ That's It!

The code automatically:
- ✅ Reads these variables
- ✅ Builds the connection string
- ✅ Connects to your database

**No connection string needed!** 🎉

## 🧪 Test

After deploying:
- Visit: `https://your-app.vercel.app/api/debug/auth`
- Should show: `"database_type": "PostgreSQL"`

