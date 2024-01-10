/******************************************************************************* 
 * Copyright (c) 2024 fxzjshm
 * This software is licensed under Mulan PubL v2.
 * You can use this software according to the terms and conditions of the Mulan PubL v2.
 * You may obtain a copy of Mulan PubL v2 at:
 *          http://license.coscl.org.cn/MulanPubL-2.0
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PubL v2 for more details.
 ******************************************************************************/

// C-style wrapper library of suzerain exprgrammar -- header file

#ifndef LIBEXPRGRAMMAR_H
#define LIBEXPRGRAMMAR_H

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

float exprgrammar_parse_float(const char* iter, const char* end);
double exprgrammar_parse_double(const char* iter, const char* end);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // LIBEXPRGRAMMAR_H
