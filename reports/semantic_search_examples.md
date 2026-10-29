# Semantic Search Benchmark Examples

> Natural language semantic search evaluation using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) on MovieLens + TMDB rich text documents.

## Summary of Capabilities
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (L2-normalized 384-dimensional dense vectors)
- **Retrieval Metric:** Cosine similarity via matrix-vector dot product
- **Filtering:** Exact genre filtering and release year interval bounds
- **Document Representation:** Title + Release Year + Curated Genres + TMDB Plot Overview + Top User Tags

---

### Query 1: "space exploration with emotional father-daughter story"
- **Category:** Theme & Emotion
- **Filters:**  *None (unrestricted catalog)*

| Rank | Title | Year | Genres | Similarity | Synopsis / Overview Snippet |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **#1** | **This Is My Father** | 1998 | Drama, Romance | `0.4945` | This Is My Father (1998). Genres: Drama, Romance.... |
| **#2** | **Dad** | 1989 | Drama | `0.4786` | Dad (1989). Genres: Drama.... |
| **#3** | **Some Mother's Son** | 1996 | Drama | `0.4730` | Some Mother's Son (1996). Genres: Drama.... |
| **#4** | **Mars Needs Moms** | 2011 | Action, Adventure, Animation | `0.4664` | Mars Needs Moms (2011). Genres: Action, Adventure, Animation, Children, Comedy, Sci-Fi, IMAX.... |
| **#5** | **Not Without My Daughter** | 1991 | Drama | `0.4569` | Not Without My Daughter (1991). Genres: Drama.... |

---

### Query 2: "lighthearted heist comedy"
- **Category:** Genre & Plot Tropes
- **Filters:**  `{'genre': 'Comedy'}`

| Rank | Title | Year | Genres | Similarity | Synopsis / Overview Snippet |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **#1** | **Maiden Heist, The** | 2009 | Comedy, Crime | `0.6826` | Maiden Heist, The (2009). Genres: Comedy, Crime.... |
| **#2** | **Tower Heist** | 2011 | Action, Comedy, Crime | `0.6523` | Tower Heist (2011). Genres: Action, Comedy, Crime.... |
| **#3** | **Schlussmacher** | 2013 | Comedy | `0.5841` | Schlussmacher (2013). Genres: Comedy.... |
| **#4** | **Semi-Pro** | 2008 | Comedy | `0.5724` | Semi-Pro (2008). Genres: Comedy.... |
| **#5** | **Stan Helsing** | 2009 | Comedy, Horror | `0.5655` | Stan Helsing (2009). Genres: Comedy, Horror.... |

---

### Query 3: "mind-bending psychological thriller with unreliable narrator"
- **Category:** Psychological / Narrative Device
- **Filters:**  `{'genre': 'Thriller'}`

| Rank | Title | Year | Genres | Similarity | Synopsis / Overview Snippet |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **#1** | **Paradox** | 2010 | Sci-Fi, Thriller | `0.5141` | Paradox (2010). Genres: Sci-Fi, Thriller.... |
| **#2** | **Guardian, The** | 1990 | Horror, Thriller | `0.5091` | Guardian, The (1990). Genres: Horror, Thriller.... |
| **#3** | **Collector, The** | 2009 | Crime, Horror, Thriller | `0.4974` | Collector, The (2009). Genres: Crime, Horror, Thriller.... |
| **#4** | **American Psycho** | 2000 | Crime, Horror, Mystery | `0.4963` | American Psycho (2000). Genres: Crime, Horror, Mystery, Thriller.... |
| **#5** | **Truth About Charlie, The** | 2002 | Mystery, Thriller | `0.4928` | Truth About Charlie, The (2002). Genres: Mystery, Thriller.... |

---

### Query 4: "animated fantasy adventure for the whole family"
- **Category:** Tone & Audience
- **Filters:**  `{'genre': 'Animation'}`

| Rank | Title | Year | Genres | Similarity | Synopsis / Overview Snippet |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **#1** | **Wild, The** | 2006 | Adventure, Animation, Children | `0.5954` | Wild, The (2006). Genres: Adventure, Animation, Children, Comedy, Fantasy.... |
| **#2** | **Monsters, Inc.** | 2001 | Adventure, Animation, Children | `0.5743` | Monsters, Inc. (2001). Genres: Adventure, Animation, Children, Comedy, Fantasy.... |
| **#3** | **Dragon Ball: Mystical Adventure (Doragon bôru: Makafushigi dai bôken)** | 1988 | Action, Adventure, Animation | `0.5647` | Dragon Ball: Mystical Adventure (Doragon bôru: Makafushigi dai bôken) (1988). Genres: Action, Advent... |
| **#4** | **Phantom Tollbooth, The** | 1970 | Adventure, Animation, Children | `0.5639` | Phantom Tollbooth, The (1970). Genres: Adventure, Animation, Children, Fantasy.... |
| **#5** | **Adventures of Rocky and Bullwinkle, The** | 2000 | Adventure, Animation, Children | `0.5631` | Adventures of Rocky and Bullwinkle, The (2000). Genres: Adventure, Animation, Children, Comedy, Fant... |

---

### Query 5: "gritty crime noir with corrupt detectives in the city"
- **Category:** Atmosphere / Style
- **Filters:**  `{'genre': 'Crime'}`

| Rank | Title | Year | Genres | Similarity | Synopsis / Overview Snippet |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **#1** | **Police Academy 6: City Under Siege** | 1989 | Comedy, Crime | `0.6434` | Police Academy 6: City Under Siege (1989). Genres: Comedy, Crime.... |
| **#2** | **Corruptor, The** | 1999 | Action, Crime, Drama | `0.6228` | Corruptor, The (1999). Genres: Action, Crime, Drama, Thriller.... |
| **#3** | **Ordinary Decent Criminal** | 2000 | Comedy, Crime | `0.6165` | Ordinary Decent Criminal (2000). Genres: Comedy, Crime.... |
| **#4** | **Perfect Crime, The (Crimen Ferpecto) (Ferpect Crime)** | 2004 | Comedy, Crime, Thriller | `0.6083` | Perfect Crime, The (Crimen Ferpecto) (Ferpect Crime) (2004). Genres: Comedy, Crime, Thriller.... |
| **#5** | **Good Thief, The** | 2002 | Crime, Drama | `0.6069` | Good Thief, The (2002). Genres: Crime, Drama.... |

---

### Query 6: "romantic comedy set in New York City"
- **Category:** Setting & Genre
- **Filters:**  `{'genre': 'Romance'}`

| Rank | Title | Year | Genres | Similarity | Synopsis / Overview Snippet |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **#1** | **Sidewalks of New York** | 2001 | Comedy, Romance | `0.7166` | Sidewalks of New York (2001). Genres: Comedy, Romance.... |
| **#2** | **Sex and the City** | 2008 | Comedy, Romance | `0.6768` | Sex and the City (2008). Genres: Comedy, Romance.... |
| **#3** | **Maid in Manhattan** | 2002 | Comedy, Romance | `0.6765` | Maid in Manhattan (2002). Genres: Comedy, Romance.... |
| **#4** | **Harlem Nights** | 1989 | Comedy, Crime, Romance | `0.6693` | Harlem Nights (1989). Genres: Comedy, Crime, Romance.... |
| **#5** | **New York, New York** | 1977 | Drama, Musical, Romance | `0.6585` | New York, New York (1977). Genres: Drama, Musical, Romance.... |

---

### Query 7: "time travel paradox and butterfly effect"
- **Category:** Sci-Fi Concept
- **Filters:**  `{'genre': 'Sci-Fi'}`

| Rank | Title | Year | Genres | Similarity | Synopsis / Overview Snippet |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **#1** | **The Butterfly Effect** | 2004 | Drama, Sci-Fi, Thriller | `0.5200` | The Butterfly Effect (2004). Genres: Drama, Sci-Fi, Thriller. Tags: alternate reality, sci-fi, scien... |
| **#2** | **Paradox** | 2010 | Sci-Fi, Thriller | `0.3822` | Paradox (2010). Genres: Sci-Fi, Thriller.... |
| **#3** | **The Butterfly Effect 3: Revelations** | 2009 | Drama, Fantasy, Sci-Fi | `0.3447` | The Butterfly Effect 3: Revelations (2009). Genres: Drama, Fantasy, Sci-Fi, Thriller.... |
| **#4** | **Frequently Asked Questions About Time Travel** | 2009 | Comedy, Sci-Fi | `0.3340` | Frequently Asked Questions About Time Travel (2009). Genres: Comedy, Sci-Fi.... |
| **#5** | **Back to the Future Part II** | 1989 | Adventure, Comedy, Sci-Fi | `0.3319` | Back to the Future Part II (1989). Genres: Adventure, Comedy, Sci-Fi. Tags: time travel.... |

---

### Query 8: "post-apocalyptic survival in a desolate wasteland"
- **Category:** World Building & Theme
- **Filters:**  *None (unrestricted catalog)*

| Rank | Title | Year | Genres | Similarity | Synopsis / Overview Snippet |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **#1** | **Just Eat It: A Food Waste Story** | 2014 | Documentary | `0.4776` | Just Eat It: A Food Waste Story (2014). Genres: Documentary. Tags: dumpster diving, sustainability.... |
| **#2** | **Winds of the Wasteland** | 1936 | Western | `0.3850` | Winds of the Wasteland (1936). Genres: Western.... |
| **#3** | **Scouts Guide to the Zombie Apocalypse** | 2015 | Action, Comedy, Horror | `0.3702` | Scouts Guide to the Zombie Apocalypse (2015). Genres: Action, Comedy, Horror.... |
| **#4** | **Postman, The** | 1997 | Action, Adventure, Drama | `0.3530` | Postman, The (1997). Genres: Action, Adventure, Drama, Sci-Fi. Tags: kevin costner, post apocalyptic... |
| **#5** | **Jonestown: The Life and Death of Peoples Temple** | 2006 | Documentary | `0.3506` | Jonestown: The Life and Death of Peoples Temple (2006). Genres: Documentary.... |

---

### Query 9: "inspiring biographical sports drama about underdog victory"
- **Category:** Narrative Arc & Mood
- **Filters:**  `{'genre': 'Drama'}`

| Rank | Title | Year | Genres | Similarity | Synopsis / Overview Snippet |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **#1** | **Sympathy for the Underdog** | 1971 | Action, Crime, Drama | `0.5289` | Sympathy for the Underdog (1971). Genres: Action, Crime, Drama.... |
| **#2** | **Hurricane, The** | 1999 | Drama | `0.5035` | Hurricane, The (1999). Genres: Drama. Tags: boxing.... |
| **#3** | **Rudy** | 1993 | Drama | `0.4585` | Rudy (1993). Genres: Drama. Tags: football.... |
| **#4** | **Golden Bowl, The** | 2000 | Drama | `0.4506` | Golden Bowl, The (2000). Genres: Drama.... |
| **#5** | **Stop-Loss** | 2008 | Drama, War | `0.4469` | Stop-Loss (2008). Genres: Drama, War.... |

---

### Query 10: "high-stakes courtroom drama with sharp dialogue"
- **Category:** Subgenre with Year Range Filter (1990-2010)
- **Filters:**  `{'year_range': (1990, 2010)}`

| Rank | Title | Year | Genres | Similarity | Synopsis / Overview Snippet |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **#1** | **Browning Version, The** | 1994 | Drama | `0.5342` | Browning Version, The (1994). Genres: Drama.... |
| **#2** | **Juror, The** | 1996 | Drama, Thriller | `0.5223` | Juror, The (1996). Genres: Drama, Thriller.... |
| **#3** | **North Country** | 2005 | Drama | `0.5179` | North Country (2005). Genres: Drama. Tags: courtroom drama, in netflix queue.... |
| **#4** | **Interview** | 2007 | Drama | `0.5162` | Interview (2007). Genres: Drama.... |
| **#5** | **King's Speech, The** | 2010 | Drama | `0.5131` | King's Speech, The (2010). Genres: Drama.... |

---
