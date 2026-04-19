# 🔌 Supabase MCP Server Setup

This guide helps you configure the Supabase MCP (Model Context Protocol) server in Cursor IDE.

## What is MCP?

MCP allows AI assistants (like me) to directly interact with your Supabase database, making it easier to:
- Query database tables
- View schema
- Run SQL queries
- Monitor database status
- Debug database issues

## Configuration

### Step 1: Get Supabase Service Role Key

1. **Go to Supabase Dashboard**
   - Visit: https://supabase.com
   - Sign in and open your project

2. **Get Service Role Key**
   - Go to **Settings** → **API**
   - Scroll to **Project API keys**
   - Copy the **`service_role`** key (NOT the `anon` key)
   - ⚠️ **Keep this secret!** It has admin access to your database

### Step 2: Configure MCP in Cursor

1. **Open Cursor Settings**
   - Press `Cmd+,` (Mac) or `Ctrl+,` (Windows/Linux)
   - Or: **Cursor** → **Settings** → **Features** → **MCP**

2. **Add MCP Server Configuration**
   
   Add this to your Cursor MCP settings:

   ```json
   {
     "mcpServers": {
       "supabase": {
         "url": "https://mcp.supabase.com/mcp",
         "apiKey": "YOUR_SERVICE_ROLE_KEY_HERE",
         "projectRef": "lgbpmgaacqawvfavtdzu"
       }
     }
   }
   ```

   **OR** if using URL-based configuration:

   ```json
   {
     "mcpServers": {
       "supabase": {
         "url": "https://mcp.supabase.com/mcp?project_ref=lgbpmgaacqawvfavtdzu",
         "apiKey": "YOUR_SERVICE_ROLE_KEY_HERE"
       }
     }
   }
   ```

### Step 3: Alternative Configuration (Environment Variables)

If MCP supports environment variables, you can also set:

```bash
SUPABASE_URL=https://lgbpmgaacqawvfavtdzu.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SUPABASE_PROJECT_REF=lgbpmgaacqawvfavtdzu
```

## Your Project Details

- **Project Reference**: `lgbpmgaacqawvfavtdzu`
- **Database Host**: `db.lgbpmgaacqawvfavtdzu.supabase.co`
- **Database Password**: `NAgyxHiwctATJwro` (for direct DB connection)
- **Service Role Key**: Get from Supabase Dashboard → Settings → API

## Verification

After configuring MCP:

1. **Restart Cursor** (if needed)
2. **Test MCP Connection**
   - Ask me: "Can you check my Supabase database tables?"
   - I should be able to query your database

## Security Notes

⚠️ **Important Security Considerations:**

1. **Service Role Key** has full admin access
   - Never commit it to git
   - Never share it publicly
   - Only use in secure environments

2. **MCP Server Access**
   - MCP server can read/write to your database
   - Use only in trusted development environments
   - Consider using read-only keys if available

3. **Environment Variables**
   - Store keys in secure environment variables
   - Use `.env` files (already gitignored)
   - Never expose in client-side code

## Troubleshooting

### MCP Server Not Connecting

1. **Check API Key**
   - Verify service role key is correct
   - Ensure it's the `service_role` key, not `anon`

2. **Check Project Reference**
   - Verify `lgbpmgaacqawvfavtdzu` is correct
   - Find it in Supabase Dashboard → Settings → General

3. **Check Cursor Settings**
   - Ensure MCP is enabled in Cursor
   - Restart Cursor after configuration changes

### Permission Errors

- Ensure you're using `service_role` key (not `anon`)
- Check Supabase project is active (not paused)
- Verify API key hasn't been rotated

## Benefits of MCP Integration

Once configured, I can:
- ✅ Query your database directly
- ✅ Check table schemas
- ✅ View user data (for debugging)
- ✅ Monitor database health
- ✅ Help debug database issues faster
- ✅ Verify migrations and schema changes

---

**Need help?** Once MCP is configured, I can help you query and manage your Supabase database directly!

