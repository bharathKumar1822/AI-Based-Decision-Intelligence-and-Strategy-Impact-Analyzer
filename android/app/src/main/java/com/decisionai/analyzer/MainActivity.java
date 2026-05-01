package com.decisionai.analyzer;

import android.app.Activity;
import android.content.Context;
import android.graphics.Color;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.os.Bundle;
import android.view.KeyEvent;
import android.view.View;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;

public class MainActivity extends Activity {

    // ──────────────────────────────────────────────────────
    // 🔗  UPDATE THIS URL after you deploy to Render.com
    // ──────────────────────────────────────────────────────
    private static final String APP_URL = "https://decision-intelligence-analyzer.onrender.com";
    // ──────────────────────────────────────────────────────

    private WebView  webView;
    private ProgressBar progressBar;
    private LinearLayout offlineLayout;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // ── Root layout ──────────────────────────────────
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.parseColor("#0d0d1a"));

        // ── Progress bar ─────────────────────────────────
        progressBar = new ProgressBar(this, null,
                android.R.attr.progressBarStyleHorizontal);
        progressBar.setMax(100);
        progressBar.setLayoutParams(new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 8));
        progressBar.setProgressTintList(
                android.content.res.ColorStateList.valueOf(
                        Color.parseColor("#6C63FF")));
        root.addView(progressBar);

        // ── WebView ──────────────────────────────────────
        webView = new WebView(this);
        webView.setLayoutParams(new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.MATCH_PARENT));
        configureWebView();
        root.addView(webView);

        // ── Offline screen ───────────────────────────────
        offlineLayout = buildOfflineLayout();
        offlineLayout.setVisibility(View.GONE);
        root.addView(offlineLayout);

        setContentView(root);

        // ── Load ─────────────────────────────────────────
        if (isConnected()) {
            webView.loadUrl(APP_URL);
        } else {
            showOffline();
        }
    }

    // ── WebView configuration ─────────────────────────────
    private void configureWebView() {
        WebSettings s = webView.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setLoadWithOverviewMode(true);
        s.setUseWideViewPort(true);
        s.setBuiltInZoomControls(false);
        s.setDisplayZoomControls(false);
        s.setCacheMode(WebSettings.LOAD_DEFAULT);
        s.setAllowFileAccess(true);
        s.setMixedContentMode(WebSettings.MIXED_CONTENT_ALWAYS_ALLOW);

        webView.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView view,
                                                    WebResourceRequest req) {
                view.loadUrl(req.getUrl().toString());
                return true;
            }

            @Override
            public void onPageFinished(WebView view, String url) {
                progressBar.setVisibility(View.GONE);
            }

            @Override
            public void onReceivedError(WebView view, int errorCode,
                                        String description, String url) {
                if (!isConnected()) showOffline();
            }
        });

        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onProgressChanged(WebView view, int progress) {
                progressBar.setVisibility(View.VISIBLE);
                progressBar.setProgress(progress);
                if (progress == 100) progressBar.setVisibility(View.GONE);
            }
        });
    }

    // ── Offline layout ────────────────────────────────────
    private LinearLayout buildOfflineLayout() {
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setGravity(android.view.Gravity.CENTER);
        layout.setBackgroundColor(Color.parseColor("#0d0d1a"));
        layout.setLayoutParams(new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.MATCH_PARENT));

        TextView icon = new TextView(this);
        icon.setText("📡");
        icon.setTextSize(64);
        icon.setGravity(android.view.Gravity.CENTER);
        layout.addView(icon);

        TextView title = new TextView(this);
        title.setText("No Internet Connection");
        title.setTextColor(Color.parseColor("#e8e8f0"));
        title.setTextSize(20);
        title.setTypeface(null, android.graphics.Typeface.BOLD);
        title.setGravity(android.view.Gravity.CENTER);
        title.setPadding(0, 16, 0, 8);
        layout.addView(title);

        TextView sub = new TextView(this);
        sub.setText("Connect to the internet and tap Retry.");
        sub.setTextColor(Color.parseColor("#9090b0"));
        sub.setTextSize(14);
        sub.setGravity(android.view.Gravity.CENTER);
        layout.addView(sub);

        TextView retry = new TextView(this);
        retry.setText("  Retry  ");
        retry.setTextColor(Color.WHITE);
        retry.setBackgroundColor(Color.parseColor("#6C63FF"));
        retry.setTextSize(16);
        retry.setTypeface(null, android.graphics.Typeface.BOLD);
        retry.setPadding(48, 24, 48, 24);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.setMargins(0, 32, 0, 0);
        retry.setLayoutParams(lp);
        retry.setOnClickListener(v -> {
            if (isConnected()) {
                offlineLayout.setVisibility(View.GONE);
                webView.setVisibility(View.VISIBLE);
                webView.loadUrl(APP_URL);
            }
        });
        layout.addView(retry);

        return layout;
    }

    private void showOffline() {
        webView.setVisibility(View.GONE);
        offlineLayout.setVisibility(View.VISIBLE);
    }

    // ── Connectivity ──────────────────────────────────────
    private boolean isConnected() {
        ConnectivityManager cm =
                (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);
        NetworkInfo ni = cm.getActiveNetworkInfo();
        return ni != null && ni.isConnected();
    }

    // ── Back button navigates WebView history ─────────────
    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        if (keyCode == KeyEvent.KEYCODE_BACK && webView.canGoBack()) {
            webView.goBack();
            return true;
        }
        return super.onKeyDown(keyCode, event);
    }

    @Override
    protected void onPause()  { super.onPause();  webView.onPause();  }
    @Override
    protected void onResume() { super.onResume(); webView.onResume(); }
    @Override
    protected void onDestroy(){ super.onDestroy(); webView.destroy(); }
}
