# False-Positive Patterns (dev fold)

Threshold 0.62. 1,156 false positives out of 58,559 predictions.

Under macro F0.5 a false merge costs roughly four times what a miss
does, and a false merge onto a true singleton takes that entity's
score from 1.0 to 0.0 outright.

| category | count | share |
|---|---|---|
| other | 499 | 43.2% |
| both_similar_overconfident | 139 | 12.0% |
| similar_address_diff_name | 129 | 11.2% |
| number_conflict | 101 | 8.7% |
| missing_address | 82 | 7.1% |
| same_name_diff_address | 78 | 6.7% |
| same_address_diff_name | 77 | 6.7% |
| similar_name_diff_address | 51 | 4.4% |

## Examples


### other

| S1 name | S1 address | candidate name | candidate address | score |
|---|---|---|---|---|
| yh tiles ltd | 2 b grant lane 3rd floor room no 47 kolk | syh tiles ltd | h no 2 b grant lane 3rd floor room no 47 | 0.989 |
| raushan industries group pvt ltd | whisperinheightbco nost marrysroadalwarp | m s raushan industries group l l p | 56 whisperinheightbco nost marrysroadalw | 0.690 |
| sarasiya travel limited | chennai raja bather street t nagar tamil | saraziya limited travel | no 52 raja bather street t nagar chennai | 0.970 |
| frick heritage mortgage | 24250 23rd avenue unit 1022 phoenix az | barrera heritage mortgage | 24250 23st ave poenix city az | 0.632 |
| baton rouge city youth church llc | 8116 brandon drive baton rouge city la | batonrougecity com | brightside view drive baton rouge city l | 0.721 |

### both_similar_overconfident

| S1 name | S1 address | candidate name | candidate address | score |
|---|---|---|---|---|
| smac media private limited | plot no c 17 sr orchid 2nd floor 201 sim | smac movie private limited | plot no c 30 sr orchid 2nd floor 201 sim | 0.780 |
| gangin services private limited | a shop near govt primary school chatrpur | ganngin services private limited | a shop near govt primary school chatrpur | 0.944 |
| lurette abrahamsen auto repair llc | 222 berry street fort wayne in | peck abrahamsen auto repair llc | 222 berry st fort wayne in | 0.764 |
| nagesh energy private limited | 5 54 a s b nagar thennandhiyalam vellore | nagosh energy private limited | h no 5 54 a s b nagar thennandhiyalam ve | 0.974 |
| kestiq northern p c | 12352 quail woods drive germantown md | cestiq northern p c | 12352 quail woods drive gerantown cdp md | 0.992 |

### similar_address_diff_name

| S1 name | S1 address | candidate name | candidate address | score |
|---|---|---|---|---|
| dream enterprises limited | aji gidc phase ii road m plot no 199 1 o | dreema entarapraaijhisa haardavera limit | rajkot phase ii road m plot no 199 1 opp | 0.645 |
| 7 point l l c | 2119 bronze bay drive missouri city tx | total pony llc | bronze bay drive missouri city tx | 0.637 |
| tarava industries | s no 128 2 p no 22 yuvraj apartments kot | taravayn industries corporation | s no 128 2 p no 22 yuvraj apartments kot | 0.973 |
| goregaon east tapes private limited | 3rd floor office no a321 desk no 7a mast | goldana prodyoosara praaiveta limiteda | office no a321 desk no 7a master mind 4  | 0.882 |
| innovative developers private limited | gat no 321 2 a p shahapur maharashtra c  | inovetiva teknolojeeja praaiveta limited | 99 c o machhindra b chougule gat no 321  | 0.920 |

### number_conflict

| S1 name | S1 address | candidate name | candidate address | score |
|---|---|---|---|---|
| heartland holdings | 6321 colebrook road henrico county va | heartland holdings holdings | 06330 colebrook road richmond va | 0.644 |
| safe trading group | 1820 kenwood avenue austin tx | safe trading group group | tx austin 1829c kenwood avenue | 0.789 |
| united academy | oh springfield 204 ardmore road | united academy group | oh 20 ardmore road springfield | 0.695 |
| thanjavur electrical private limited | ward no 1 nangli road sohna gurgaon hary | private thanjavur electrical group limit | plot 892 ward no 10 nangli road sohna gu | 0.740 |
| kiser empire interactive llc | 3400 mission arch drive roswell nm | kiser empire interactive greater llc | 340 mission arch drive roswell nm | 0.627 |

### missing_address

| S1 name | S1 address | candidate name | candidate address | score |
|---|---|---|---|---|
| daria n rios ph d m d p c llc | 1160 bellflower way los banos ca | daria n rios ph d m d p c |  | 0.722 |
| india vivek holding private limited | 4 batteary lane rajpur road civil lane n | india vivek holding private |  | 0.819 |
| novyn municipals llc | 3385 2600 fillmore ut | novyn municipals llc |  | 0.810 |
| duvall acquisitions llc | 1803 coxemoor place asheboro nc | duvall acquisitions |  | 0.805 |
| wang nevitt ingredients | 2755 aguila drive wickenburg az | the wang nevitt ingredients |  | 0.689 |

### same_name_diff_address

| S1 name | S1 address | candidate name | candidate address | score |
|---|---|---|---|---|
| dominica s brewing inc | 405 a street packwood ia | dominica s brewing inc | 407 a street packwood ia | 0.975 |
| professional digital digital inc | 16059 122 hi hat ky | professional digital digital inc | 16061 122 hi hat ky | 0.973 |
| dermatology care associates of albuquerq | 1208 juan tabo boulevard albuquerque nm | dermatology care associates of albuquerq | regina road albuquerque nm | 0.789 |
| dhanalakshmi forex private limited | 11 r n mukherjee road nilhat house 10th  | dhanalakshmi forex private limited | 11th floor industry house 10 camac stree | 0.731 |
| new delhi services private limited | h no g 70 f f a f enclave 1 okhla new de | new delhi services private limited | b 1 g f m n b 1 g f press enclave malviy | 0.897 |
