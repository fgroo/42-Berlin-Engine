/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   fluid_get.c                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/19 19:40:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/19 19:40:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

/*
** ===========================================================================
** FLUID-GET: The Knowledge Package Manager
** ===========================================================================
** Decentralized distribution for .fluid capsules.
** 
** Commands:
**   update    - Fetch latest repository index
**   list      - Show available packages
**   install   - Download and verify a package
**   search    - Find packages by keyword
**
** Repository Format (index.fl):
**   # domain|version|base_hash|url|signature
**   coding/c|1.0|0xDEADBEEF|https://hub.42.berlin/c.fluid|SIG_ABC
** ===========================================================================
*/

#include "../fluid/fluid_spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

#define DEFAULT_REPO "file:///home/nix/42-Berlin-Engine/registry/index.fl"
#define FLUID_DIR "./fluids/"
#define INDEX_CACHE "/tmp/fluid_index.fl"
#define MAX_LINE 1024

/*
** Ensure the fluids directory exists
*/
static void	ensure_fluid_dir(void)
{
	struct stat	st;

	if (stat(FLUID_DIR, &st) == -1)
		mkdir(FLUID_DIR, 0755);
}

/*
** Download a file using curl (or cp for file:// URLs)
*/
static int	download_file(const char *url, const char *dest)
{
	char	cmd[2048];

	printf("[NET] Fetching %s...\n", url);
	/* Handle file:// URLs with cp */
	if (strncmp(url, "file://", 7) == 0)
	{
		snprintf(cmd, sizeof(cmd), "cp '%s' '%s' 2>/dev/null", url + 7, dest);
	}
	else
	{
		snprintf(cmd, sizeof(cmd), 
			"curl -s -L -f --connect-timeout 10 -o '%s' '%s' 2>/dev/null", 
			dest, url);
	}
	if (system(cmd) != 0)
	{
		fprintf(stderr, "[NET] Download failed!\n");
		return (0);
	}
	return (1);
}

/*
** Verify a downloaded .fluid file
*/
static int	verify_fluid(const char *filepath, const char *expected_domain)
{
	FILE			*f;
	t_fluid_header	h;

	f = fopen(filepath, "rb");
	if (!f)
	{
		fprintf(stderr, "[VERIFY] Cannot open file\n");
		return (0);
	}
	if (fread(&h, sizeof(h), 1, f) != 1)
	{
		fclose(f);
		fprintf(stderr, "[VERIFY] Cannot read header\n");
		return (0);
	}
	fclose(f);
	/* Validate magic */
	if (memcmp(h.magic, FLUID_MAGIC, 4) != 0)
	{
		fprintf(stderr, "[VERIFY] Invalid magic bytes!\n");
		return (0);
	}
	printf("[VERIFY] ✓ Valid FLv%d capsule\n", h.version);
	printf("[VERIFY] ✓ Domain: %s\n", h.domain[0] ? h.domain : "(unset)");
	printf("[VERIFY] ✓ Author: %s\n", h.author[0] ? h.author : "(unknown)");
	printf("[VERIFY] ✓ Entries: %u patterns\n", h.n_entries);
	(void)expected_domain;  /* TODO: Match against registry entry */
	return (1);
}

/*
** Print banner
*/
static void	print_banner(void)
{
	printf("\n");
	printf("╔══════════════════════════════════════════════════════════════╗\n");
	printf("║           FLUID-GET: The Knowledge Package Manager           ║\n");
	printf("╚══════════════════════════════════════════════════════════════╝\n");
	printf("\n");
}

/*
** Handle 'update' command - fetch repository index
*/
static int	cmd_update(void)
{
	printf("[UPDATE] Fetching repository index...\n");
	if (!download_file(DEFAULT_REPO, INDEX_CACHE))
	{
		fprintf(stderr, "[ERROR] Could not fetch repository index.\n");
		fprintf(stderr, "[HINT] Check network or repository URL.\n");
		return (1);
	}
	printf("[UPDATE] ✓ Repository index updated.\n");
	printf("[UPDATE] Run 'fluid-get list' to see available packages.\n");
	return (0);
}

/*
** Handle 'list' command - show available packages
*/
static int	cmd_list(void)
{
	FILE	*f;
	char	line[MAX_LINE];
	char	*domain;
	char	*ver;
	char	*hash;
	char	*url;
	int		count;

	f = fopen(INDEX_CACHE, "r");
	if (!f)
	{
		fprintf(stderr, "[ERROR] No cached index. Run 'fluid-get update' first.\n");
		return (1);
	}
	printf("AVAILABLE PACKAGES:\n");
	printf("═══════════════════════════════════════════════════════════════\n");
	printf("%-25s │ %-8s │ %s\n", "DOMAIN", "VERSION", "SOURCE");
	printf("───────────────────────────┼──────────┼────────────────────────\n");
	count = 0;
	while (fgets(line, sizeof(line), f))
	{
		if (line[0] == '#' || line[0] == '\n')
			continue;
		line[strcspn(line, "\r\n")] = 0;
		domain = strtok(line, "|");
		ver = strtok(NULL, "|");
		hash = strtok(NULL, "|");
		url = strtok(NULL, "|");
		if (domain && ver && url)
		{
			/* Truncate long URLs */
			char url_short[32];
			size_t url_len = strlen(url);
			if (url_len > 24)
			{
				memcpy(url_short, url, 21);
				url_short[21] = '\0';
				strcat(url_short, "...");
			}
			else
			{
				strcpy(url_short, url);
			}
			printf("%-25s │ %-8s │ %s\n", domain, ver, url_short);
			count++;
		}
		(void)hash;
	}
	fclose(f);
	printf("═══════════════════════════════════════════════════════════════\n");
	printf("Total: %d packages available\n\n", count);
	return (0);
}

/*
** Handle 'search' command - find packages by keyword
*/
static int	cmd_search(const char *keyword)
{
	FILE	*f;
	char	line[MAX_LINE];
	char	linecopy[MAX_LINE];
	int		found;

	f = fopen(INDEX_CACHE, "r");
	if (!f)
	{
		fprintf(stderr, "[ERROR] No cached index. Run 'fluid-get update' first.\n");
		return (1);
	}
	printf("Searching for '%s'...\n\n", keyword);
	found = 0;
	while (fgets(line, sizeof(line), f))
	{
		if (line[0] == '#' || line[0] == '\n')
			continue;
		strcpy(linecopy, line);
		if (strstr(linecopy, keyword))
		{
			linecopy[strcspn(linecopy, "\r\n")] = 0;
			char *domain = strtok(linecopy, "|");
			char *ver = strtok(NULL, "|");
			if (domain && ver)
			{
				printf("  → %s (v%s)\n", domain, ver);
				found++;
			}
		}
	}
	fclose(f);
	if (found == 0)
		printf("No packages found matching '%s'\n", keyword);
	else
		printf("\n%d package(s) found.\n", found);
	return (0);
}

/*
** Handle 'install' command - download and install a package
*/
static int	cmd_install(const char *target_domain)
{
	FILE	*f;
	char	line[MAX_LINE];
	char	found_url[512];
	char	dest_path[512];
	char	safe_name[256];
	char	*p;

	f = fopen(INDEX_CACHE, "r");
	if (!f)
	{
		fprintf(stderr, "[ERROR] No cached index. Run 'fluid-get update' first.\n");
		return (1);
	}
	found_url[0] = '\0';
	while (fgets(line, sizeof(line), f))
	{
		if (line[0] == '#' || line[0] == '\n')
			continue;
		char tmp[MAX_LINE];
		strcpy(tmp, line);
		tmp[strcspn(tmp, "\r\n")] = 0;
		char *domain = strtok(tmp, "|");
		char *ver = strtok(NULL, "|");
		char *hash = strtok(NULL, "|");
		char *url = strtok(NULL, "|");
		(void)ver;
		(void)hash;
		if (domain && url && strcmp(domain, target_domain) == 0)
		{
			strncpy(found_url, url, sizeof(found_url) - 1);
			break;
		}
	}
	fclose(f);
	if (found_url[0] == '\0')
	{
		fprintf(stderr, "[ERROR] Package '%s' not found in registry.\n", target_domain);
		fprintf(stderr, "[HINT] Run 'fluid-get list' to see available packages.\n");
		return (1);
	}
	/* Create safe filename (replace / with _) */
	strncpy(safe_name, target_domain, sizeof(safe_name) - 1);
	for (p = safe_name; *p; p++)
	{
		if (*p == '/')
			*p = '_';
	}
	snprintf(dest_path, sizeof(dest_path), "%s%s.fluid", FLUID_DIR, safe_name);
	printf("[INSTALL] Installing %s...\n", target_domain);
	if (!download_file(found_url, dest_path))
	{
		fprintf(stderr, "[ERROR] Download failed!\n");
		return (1);
	}
	if (!verify_fluid(dest_path, target_domain))
	{
		fprintf(stderr, "[ERROR] Verification failed! Removing corrupted file.\n");
		unlink(dest_path);
		return (1);
	}
	printf("\n");
	printf("═══════════════════════════════════════════════════════════════\n");
	printf("       ✓ INSTALLED: %s → %s\n", target_domain, dest_path);
	printf("═══════════════════════════════════════════════════════════════\n");
	printf("\n");
	printf("Load in engine with: ./42-engine -f %s\n\n", dest_path);
	return (0);
}

/*
** Print usage
*/
static void	print_usage(const char *prog)
{
	printf("Usage: %s <command> [args]\n\n", prog);
	printf("Commands:\n");
	printf("  update            Fetch latest package index\n");
	printf("  list              Show available packages\n");
	printf("  search <keyword>  Find packages by name\n");
	printf("  install <domain>  Download and install a package\n");
	printf("\n");
	printf("Examples:\n");
	printf("  %s update\n", prog);
	printf("  %s list\n", prog);
	printf("  %s install coding/c\n", prog);
	printf("\n");
}

int	main(int argc, char **argv)
{
	if (argc < 2)
	{
		print_banner();
		print_usage(argv[0]);
		return (1);
	}
	ensure_fluid_dir();
	print_banner();
	if (strcmp(argv[1], "update") == 0)
		return (cmd_update());
	else if (strcmp(argv[1], "list") == 0)
		return (cmd_list());
	else if (strcmp(argv[1], "search") == 0)
	{
		if (argc < 3)
		{
			fprintf(stderr, "Usage: %s search <keyword>\n", argv[0]);
			return (1);
		}
		return (cmd_search(argv[2]));
	}
	else if (strcmp(argv[1], "install") == 0)
	{
		if (argc < 3)
		{
			fprintf(stderr, "Usage: %s install <domain>\n", argv[0]);
			return (1);
		}
		return (cmd_install(argv[2]));
	}
	else
	{
		fprintf(stderr, "Unknown command: %s\n", argv[1]);
		print_usage(argv[0]);
		return (1);
	}
}
