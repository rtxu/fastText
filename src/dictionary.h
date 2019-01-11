/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <istream>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "args.h"
#include "real.h"

namespace fasttext {

typedef int32_t id_type;
enum class entry_type : int8_t { word = 0, label = 1 };

struct entry {
  std::string word;
  int64_t count;
  entry_type type;
  std::vector<int32_t> subwords;
};

/*
 * 词典的构造过程如下:
 *  1. 从用户输入中逐一读出单词 w
 *      1.1 查 hash 表，获取 w 在 hash 表中的位置，记为 H(w)
 *          冲突解决：查找到位置可能已有的其他单词，如果冲突则顺延，直到找到空白位置，
 *  2. 如果 H(w) 的位置仍无单词占用，则发现新词 w，按照新词发现的顺序，按序记录在词典中
 *  3. 如果 H(w) 的位置已有词，则累加词频
 *
 * 关键数据结构：
 *  - find(w)：在 hash 表中查找 w 对应的 H(w)
 *  - word2int_[H(w)]: 将 H(w) 映射为 w 在词典中的位置
 *  _ words_[i]: 词典中位置为 i 的词条
 */
class Dictionary {
 protected:
  // hash 表大小
  static const int32_t MAX_VOCAB_SIZE = 30000000;
  // 单行文本最大单词数
  static const int32_t MAX_LINE_SIZE = 1024;

  int32_t find(const std::string&) const;
  int32_t find(const std::string&, uint32_t h) const;
  void initTableDiscard();
  void initNgrams();
  void reset(std::istream&) const;
  void pushHash(std::vector<int32_t>&, int32_t) const;
  void addSubwords(std::vector<int32_t>&, const std::string&, int32_t) const;

  std::shared_ptr<Args> args_;
  std::vector<int32_t> word2int_;
  std::vector<entry> words_;

  // cbow 和 skipgram 模型使用，用于对高频词降权
  std::vector<real> pdiscard_;

  // 词典大小，词典里既包含 word 也包含 label
  int32_t size_;
  int32_t nwords_;
  int32_t nlabels_;
  // input 的 token 数
  int64_t ntokens_;

  int64_t pruneidx_size_;
  std::unordered_map<int32_t, int32_t> pruneidx_;
  void addWordNgrams(
      std::vector<int32_t>& line,
      const std::vector<int32_t>& hashes,
      int32_t n) const;

 public:
  static const std::string EOS;
  static const std::string BOW;
  static const std::string EOW;

  explicit Dictionary(std::shared_ptr<Args>);
  explicit Dictionary(std::shared_ptr<Args>, std::istream&);
  int32_t nwords() const;
  int32_t nlabels() const;
  int64_t ntokens() const;
  int32_t getId(const std::string&) const;
  int32_t getId(const std::string&, uint32_t h) const;
  entry_type getType(int32_t) const;
  entry_type getType(const std::string&) const;
  bool discard(int32_t, real) const;
  std::string getWord(int32_t) const;
  const std::vector<int32_t>& getSubwords(int32_t) const;
  const std::vector<int32_t> getSubwords(const std::string&) const;
  void getSubwords(
      const std::string&,
      std::vector<int32_t>&,
      std::vector<std::string>&) const;
  void computeSubwords(
      const std::string&,
      std::vector<int32_t>&,
      std::vector<std::string>* substrings = nullptr) const;
  uint32_t hash(const std::string& str) const;
  void add(const std::string&);
  bool readWord(std::istream&, std::string&) const;
  // 处理用户输入，构造词典
  void readFromFile(std::istream&);
  std::string getLabel(int32_t) const;
  void save(std::ostream&) const;
  void load(std::istream&);
  std::vector<int64_t> getCounts(entry_type) const;
  int32_t getLine(std::istream&, std::vector<int32_t>&, std::vector<int32_t>&)
      const;
  int32_t getLine(std::istream&, std::vector<int32_t>&, std::minstd_rand&)
      const;
  void threshold(int64_t, int64_t);
  void prune(std::vector<int32_t>&);
  bool isPruned() {
    return pruneidx_size_ >= 0;
  }
  void dump(std::ostream&) const;
  void init();
};

} // namespace fasttext
