// Command agentic_fix iteratively fixes Go build errors caused by LXD client
// SDK changes using an AI model via the OpenRouter API.
//
// It performs the following loop (up to MAX_ITERATIONS times):
//  1. Parse build errors to identify affected source files
//  2. Send errors + LXD client diff + source files to the AI model
//  3. Apply the returned file edits
//  4. Rebuild and check if errors are resolved
//
// Environment variables:
//
//	OPENROUTER_API_KEY   - API key for OpenRouter
//	OPENROUTER_MODEL     - Model identifier (e.g. anthropic/claude-sonnet-4)
//	BUILD_ERRORS_FILE    - Path to file containing initial build errors
//	LXD_CLIENT_DIFF_FILE - Path to file containing the LXD client diff
//	MAX_ITERATIONS       - Maximum fix attempts (default: 5)
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

const openRouterAPIURL = "https://openrouter.ai/api/v1/chat/completions"

const systemPrompt = `You are a Go developer working on the Terraform Provider for LXD ` +
	`(github.com/terraform-lxd/terraform-provider-lxd). The LXD client SDK ` +
	`(github.com/canonical/lxd) has been updated and the provider code no longer compiles.

Your task is to fix the Go build errors caused by API changes in the LXD client package. You will be given:
- The diff of what changed in the LXD client SDK
- The current build errors
- The content of each affected source file

Rules:
1. Only modify files that have build errors — do not refactor unrelated code.
2. Preserve existing logic and behavior; only adapt to the new client API.
3. If a type or function was renamed, update all references.
4. If a function signature changed, update call sites to match.
5. If a type gained or lost fields, update struct literals accordingly.
6. Do not add new features or change test logic.

Respond with ONLY a JSON array of file edits. Each element must have:
- "file": the relative file path (e.g. "internal/network/resource_network.go")
- "content": the complete updated file content

Example response format:
[
  {
    "file": "internal/network/resource_network.go",
    "content": "package network\n\nimport (...)\n..."
  }
]

If no changes are needed for a file, omit it from the array.
Return ONLY the JSON array — no markdown fences, no explanation.`

// fileEdit represents a single file modification returned by the AI.
type fileEdit struct {
	File    string `json:"file"`
	Content string `json:"content"`
}

// chatMessage represents an OpenRouter chat message.
type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// chatRequest is the OpenRouter API request payload.
type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Temperature float64       `json:"temperature"`
	MaxTokens   int           `json:"max_tokens"`
}

// chatResponse is the relevant portion of the OpenRouter API response.
type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
		Code    int    `json:"code"`
	} `json:"error"`
}

func main() {
	os.Exit(run())
}

func run() int {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	model := os.Getenv("OPENROUTER_MODEL")
	buildErrorsFile := envOrDefault("BUILD_ERRORS_FILE", "build-errors.txt")
	lxdDiffFile := envOrDefault("LXD_CLIENT_DIFF_FILE", "lxd-client-diff.txt")
	maxIterations := envOrDefaultInt("MAX_ITERATIONS", 5)

	if apiKey == "" {
		fmt.Println("ERROR: OPENROUTER_API_KEY is not set.")
		return 1
	}

	if model == "" {
		fmt.Println("ERROR: OPENROUTER_MODEL is not set.")
		return 1
	}

	fmt.Printf("Model: %s\n", model)
	fmt.Printf("Max iterations: %d\n", maxIterations)

	// Connectivity check: verify we can reach the OpenRouter API.
	fmt.Println("\nChecking OpenRouter API connectivity...")
	if err := checkConnectivity(apiKey); err != nil {
		fmt.Printf("ERROR: OpenRouter API connectivity check failed: %v\n", err)
		return 1
	}

	fmt.Println("OpenRouter API is reachable.")

	buildErrors, err := readFile(buildErrorsFile)
	if err != nil {
		fmt.Printf("ERROR: Failed to read build errors file: %v\n", err)
		return 1
	}

	lxdDiff, _ := readFile(lxdDiffFile) // OK if missing.

	var summaryParts []string
	summaryParts = append(summaryParts, "## AI Fix Summary\n")

	allModified := make(map[string]bool)
	finalSuccess := false

	// Keep conversation history across iterations so the AI can learn from
	// previous attempts and avoid repeating the same mistakes.
	messages := []chatMessage{
		{Role: "system", Content: systemPrompt},
	}

	for iteration := 1; iteration <= maxIterations; iteration++ {
		fmt.Printf("\n%s\n", strings.Repeat("=", 60))
		fmt.Printf("Iteration %d/%d\n", iteration, maxIterations)
		fmt.Printf("%s\n", strings.Repeat("=", 60))

		affectedFiles := extractAffectedFiles(buildErrors)
		if len(affectedFiles) == 0 {
			fmt.Println("No affected Go files found in build errors.")
			fmt.Println("Build errors may not be file-specific. Dumping errors:")
			fmt.Println(buildErrors)
			break
		}

		fmt.Printf("Affected files (%d):\n", len(affectedFiles))
		for _, f := range affectedFiles {
			fmt.Printf("  - %s\n", f)
		}

		userPrompt := buildUserPrompt(buildErrors, lxdDiff, affectedFiles, iteration)

		promptSize := len(userPrompt)
		fmt.Printf("Prompt size: %d chars (~%d tokens)\n", promptSize, promptSize/4)

		messages = append(messages, chatMessage{Role: "user", Content: userPrompt})

		fmt.Printf("\nCalling OpenRouter (%s)...\n", model)
		response, err := callOpenRouter(model, apiKey, messages)
		if err != nil {
			fmt.Printf("  ✗ API call failed: %v\n", err)
			summaryParts = append(summaryParts, fmt.Sprintf("### Iteration %d\n- API call failed: %v\n", iteration, err))

			// Remove the last user message so we can retry cleanly.
			messages = messages[:len(messages)-1]
			continue
		}

		fmt.Printf("Response size: %d chars\n", len(response))

		// Add AI response to conversation history.
		messages = append(messages, chatMessage{Role: "assistant", Content: response})

		edits := parseFileEdits(response)
		if len(edits) == 0 {
			fmt.Println("  ✗ No valid edits returned by AI.")
			summaryParts = append(summaryParts, fmt.Sprintf("### Iteration %d\n- No valid edits returned\n", iteration))
			continue
		}

		fmt.Printf("\nApplying %d edit(s):\n", len(edits))
		modified := applyEdits(edits)
		for _, f := range modified {
			allModified[f] = true
		}

		summaryParts = append(summaryParts, fmt.Sprintf("### Iteration %d\n- Modified: %s\n", iteration, strings.Join(modified, ", ")))

		fmt.Println("\nRebuilding...")
		exitCode, buildOutput := runBuild()

		if exitCode == 0 {
			fmt.Println("✅ Build succeeded!")
			summaryParts = append(summaryParts, "- **Result: Build passed** ✅\n")
			finalSuccess = true
			break
		}

		fmt.Printf("Build still failing (exit code %d).\n", exitCode)
		buildErrors = buildOutput

		// Update the build errors file for the workflow.
		_ = writeFile(buildErrorsFile, buildErrors)
		summaryParts = append(summaryParts, "- Result: Build still failing\n")
	}

	// Write fix summary.
	modifiedList := sortedKeys(allModified)
	summaryParts = append(summaryParts, fmt.Sprintf("\n**Files modified:** %d\n", len(modifiedList)))
	for _, f := range modifiedList {
		summaryParts = append(summaryParts, fmt.Sprintf("- `%s`\n", f))
	}

	_ = writeFile("fix-summary.md", strings.Join(summaryParts, ""))

	if finalSuccess {
		fmt.Println("\n✅ AI fix completed successfully.")
		return 0
	}

	fmt.Println("\n⚠️  AI fix incomplete after all iterations.")
	return 1
}

// checkConnectivity verifies the OpenRouter API is reachable by sending a
// minimal request. We intentionally send a tiny prompt so costs are negligible.
func checkConnectivity(apiKey string) error {
	reqBody := chatRequest{
		Model: "openai/gpt-4.1-nano",
		Messages: []chatMessage{
			{Role: "user", Content: "Reply with OK"},
		},
		Temperature: 0,
		MaxTokens:   5,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, openRouterAPIURL, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}

	defer resp.Body.Close()

	if resp.StatusCode == http.StatusUnauthorized {
		return fmt.Errorf("authentication failed (HTTP 401) — check OPENROUTER_API_KEY")
	}

	if resp.StatusCode >= 500 {
		return fmt.Errorf("server error (HTTP %d)", resp.StatusCode)
	}

	// Any 2xx or 4xx (other than 401) means the API is reachable.
	return nil
}

// callOpenRouter sends messages to the OpenRouter chat completions API.
func callOpenRouter(model, apiKey string, messages []chatMessage) (string, error) {
	reqBody := chatRequest{
		Model:       model,
		Messages:    messages,
		Temperature: 0.0,
		MaxTokens:   64000,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, openRouterAPIURL, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("HTTP-Referer", "https://github.com/lxd-clientbot/terraform-provider-lxd")
	req.Header.Set("X-Title", "LXD Terraform Provider - Client Update Bot")

	client := &http.Client{Timeout: 5 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("request failed: %w", err)
	}

	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(respBody[:min(len(respBody), 500)]))
	}

	var chatResp chatResponse
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return "", fmt.Errorf("unmarshal response: %w", err)
	}

	if chatResp.Error != nil {
		return "", fmt.Errorf("API error: %s", chatResp.Error.Message)
	}

	if len(chatResp.Choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}

	return chatResp.Choices[0].Message.Content, nil
}

// extractAffectedFiles parses Go build error output to find unique file paths.
func extractAffectedFiles(buildErrors string) []string {
	pattern := regexp.MustCompile(`(?m)^(?:\./)?([^\s:]+\.go):\d+`)
	matches := pattern.FindAllStringSubmatch(buildErrors, -1)

	seen := make(map[string]bool)
	var files []string

	for _, m := range matches {
		fp := m[1]
		if seen[fp] {
			continue
		}

		if _, err := os.Stat(fp); err != nil {
			continue
		}

		seen[fp] = true
		files = append(files, fp)
	}

	sort.Strings(files)
	return files
}

// runBuild executes "go build ./..." and returns the exit code and output.
func runBuild() (int, string) {
	cmd := exec.Command("go", "build", "./...")

	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf

	err := cmd.Run()

	output := buf.String()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return exitErr.ExitCode(), output
		}

		return 1, output
	}

	return 0, output
}

// buildUserPrompt constructs the user prompt for the AI model.
func buildUserPrompt(buildErrors, lxdDiff string, affectedFiles []string, iteration int) string {
	var b strings.Builder

	fmt.Fprintf(&b, "## Iteration %d\n\n", iteration)

	b.WriteString("## LXD Client Diff\n")
	if len(lxdDiff) > 50000 {
		b.WriteString(lxdDiff[:50000])
		b.WriteString("\n... (diff truncated) ...\n")
	} else {
		b.WriteString(lxdDiff)
	}

	b.WriteString("\n## Build Errors\n")
	b.WriteString(buildErrors)

	b.WriteString("\n## Affected Source Files\n")
	for _, fp := range affectedFiles {
		content, err := readFile(fp)
		if err != nil {
			continue
		}

		fmt.Fprintf(&b, "\n### %s\n```go\n%s\n```\n", fp, content)
	}

	b.WriteString("\nFix the build errors above. Return ONLY a JSON array of file edits.")

	return b.String()
}

// parseFileEdits extracts file edits from an AI response. Handles optional
// markdown code fences around the JSON.
func parseFileEdits(response string) []fileEdit {
	cleaned := strings.TrimSpace(response)

	// Strip markdown code fences if present.
	cleaned = regexp.MustCompile(`(?s)^` + "```" + `(?:json)?\s*\n?`).ReplaceAllString(cleaned, "")
	cleaned = regexp.MustCompile(`(?s)\n?` + "```" + `\s*$`).ReplaceAllString(cleaned, "")
	cleaned = strings.TrimSpace(cleaned)

	var edits []fileEdit
	if err := json.Unmarshal([]byte(cleaned), &edits); err != nil {
		fmt.Printf("  ✗ Failed to parse AI response as JSON: %v\n", err)
		preview := cleaned
		if len(preview) > 500 {
			preview = preview[:500]
		}

		fmt.Printf("  Response preview: %s\n", preview)
		return nil
	}

	var valid []fileEdit
	for _, e := range edits {
		if e.File == "" || e.Content == "" {
			fmt.Printf("  ✗ Skipping malformed edit entry (empty file or content)\n")
			continue
		}

		valid = append(valid, e)
	}

	return valid
}

// applyEdits writes file edits to disk and returns the list of modified paths.
func applyEdits(edits []fileEdit) []string {
	var modified []string

	for _, edit := range edits {
		if _, err := os.Stat(edit.File); os.IsNotExist(err) {
			fmt.Printf("  ⚠ Skipping non-existent file: %s\n", edit.File)
			continue
		}

		if err := writeFile(edit.File, edit.Content); err != nil {
			fmt.Printf("  ✗ Failed to write %s: %v\n", edit.File, err)
			continue
		}

		modified = append(modified, edit.File)
		fmt.Printf("  ✓ Updated %s\n", edit.File)
	}

	return modified
}

// readFile reads a file and returns its content.
func readFile(path string) (string, error) {
	data, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return "", err
	}

	return string(data), nil
}

// writeFile writes content to a file.
func writeFile(path, content string) error {
	return os.WriteFile(filepath.Clean(path), []byte(content), 0644)
}

func envOrDefault(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}

	return fallback
}

func envOrDefaultInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}

	return fallback
}

func sortedKeys(m map[string]bool) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}

	sort.Strings(keys)
	return keys
}
