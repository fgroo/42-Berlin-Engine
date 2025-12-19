/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   fluid_io.h                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgroo <fgroo@student.42berlin.de>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/12/19 19:00:00 by fgroo             #+#    #+#             */
/*   Updated: 2025/12/19 19:00:00 by fgroo            ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef FLUID_IO_H
# define FLUID_IO_H

# include "fluid_spec.h"

/*
** Header creation and modification
*/
t_fluid_header	fluid_create_header(const char *domain, const char *author,
					uint64_t base_hash);
void			fluid_set_description(t_fluid_header *h, const char *desc);

/*
** File I/O
*/
int				fluid_write_file(const char *filename, t_fluid_header *header,
					t_fluid_entry *entries);
int				fluid_read_header(const char *filename,
					t_fluid_header *out_header);
int				fluid_read_file(const char *filename, t_fluid_header *out_header,
					t_fluid_entry **out_entries);

/*
** Validation
*/
int				fluid_validate_compatibility(const t_fluid_header *header,
					uint64_t current_model_hash);

/*
** Error handling
*/
const char		*fluid_strerror(int err);

#endif
