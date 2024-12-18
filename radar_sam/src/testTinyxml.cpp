#include "../tinyxml2/tinyxml2.h"
#include <iostream>
#include <string>

using namespace tinyxml2;
using namespace std;
int main()
{
  XMLDocument doc;
  //XMLError errXml = doc.LoadFile("/media/zhaoyz/code_ws/radar_slam_ws2/src/radar_sam_ros2/radar_sam/config/semantic_map.xml");
  XMLError errXml = doc.LoadFile("/media/zhaoyz/code_ws/radar_slam_ws2/gpal_semantic_map.xml");
  if (XML_SUCCESS != errXml)
  {
    cout << "load failed!" << doc.ErrorIDToName(errXml) << endl;
    return -1;
  }
  XMLElement *elmtRoot = doc.RootElement();
  cout << "root:" << elmtRoot->Value() << endl;
  XMLElement *elmt = elmtRoot->FirstChildElement();

  string attribute;
  string value;
  while (elmt)
  {
    XMLElement *category = elmt->FirstChildElement("category");
    string str;
    if (category != 0)
    {
      const char *label = category->GetText();
      cout << label << endl;
      string s(label);
      str = s;
    }

    if (str == "parkingslot")
    {
      XMLElement *type = elmt->FirstChildElement("type");
      XMLElement *id = elmt->FirstChildElement("id");
      XMLElement *pose = elmt->FirstChildElement("pose");
      XMLElement *pa = pose->FirstChildElement("pa");
      XMLElement *pax = pa->FirstChildElement("x");
      XMLElement *pay = pa->FirstChildElement("y");

      XMLElement *pb = pose->FirstChildElement("pb");
      XMLElement *pbx = pb->FirstChildElement("x");
      XMLElement *pby = pb->FirstChildElement("y");

      XMLElement *pc = pose->FirstChildElement("pc");
      XMLElement *pcx = pc->FirstChildElement("x");
      XMLElement *pcy = pc->FirstChildElement("y");

      XMLElement *pd = pose->FirstChildElement("pd");
      XMLElement *pdx = pd->FirstChildElement("x");
      XMLElement *pdy = pd->FirstChildElement("y");

      XMLElement *score = elmt->FirstChildElement("score");
      XMLElement *occ = elmt->FirstChildElement("occ");

      // string category, type, id;
      cout << category->GetText() << endl;
      cout << type->GetText() << endl;
      cout << id->GetText() << endl;
      double ax, ay, bx, by, cx, cy, dx, dy;
      pax->QueryDoubleText(&ax);
      pay->QueryDoubleText(&ay);
      pbx->QueryDoubleText(&bx);
      pby->QueryDoubleText(&by);
      pcx->QueryDoubleText(&cx);
      pcy->QueryDoubleText(&cy);
      pdx->QueryDoubleText(&dx);
      pdy->QueryDoubleText(&dy);
      cout << ax << " " << ay << endl;
      cout << bx << " " << by << endl;
      cout << cx << " " << cy << endl;
      cout << dx << " " << dy << endl;

      double s;
      score->QueryDoubleText(&s);
      int o;
      occ->QueryIntText(&o);
      cout << s << endl;
      cout << o << endl;
    }

    elmt = elmt->NextSiblingElement();
  }

  // test write xml
  XMLDocument *xmlDoc = new XMLDocument();
  XMLNode *pRoot = xmlDoc->NewElement("gpalsemanticmap");
  xmlDoc->InsertFirstChild(pRoot);

  XMLElement *pElement = xmlDoc->NewElement("element");
  XMLElement *category = xmlDoc->NewElement("category");
  category->SetText("parkingslot");
  pElement->InsertEndChild(category);

  XMLElement *type = xmlDoc->NewElement("type");
  type->SetText("水平");
  pElement->InsertEndChild(type);

  XMLElement *id = xmlDoc->NewElement("id");
  id->SetText("000");
  pElement->InsertEndChild(id);


  XMLElement *pose = xmlDoc->NewElement("pose");
  XMLElement *pa = xmlDoc->NewElement("pa");
  XMLElement *pax = xmlDoc->NewElement("x");
  pax->SetText(0.1);
  pa->InsertEndChild(pax);
  XMLElement *pay = xmlDoc->NewElement("y");
  pay->SetText(0.2);
  pa->InsertEndChild(pay);
  pose->InsertEndChild(pa);

  XMLElement *pb = xmlDoc->NewElement("pb");
  XMLElement *pbx = xmlDoc->NewElement("x");
  pbx->SetText(0.3);
  pb->InsertEndChild(pbx);
  XMLElement *pby = xmlDoc->NewElement("y");
  pby->SetText(0.4);
  pb->InsertEndChild(pby);
  pose->InsertEndChild(pb);

  XMLElement *pc = xmlDoc->NewElement("pc");
  XMLElement *pcx = xmlDoc->NewElement("x");
  pcx->SetText(0.5);
  pc->InsertEndChild(pcx);
  XMLElement *pcy = xmlDoc->NewElement("y");
  pcy->SetText(0.6);
  pc->InsertEndChild(pcy);
  pose->InsertEndChild(pc);

  XMLElement *pd = xmlDoc->NewElement("pd");
  XMLElement *pdx = xmlDoc->NewElement("x");
  pdx->SetText(0.7);
  pd->InsertEndChild(pdx);
  XMLElement *pdy = xmlDoc->NewElement("y");
  pdy->SetText(0.8);
  pd->InsertEndChild(pdy);
  pose->InsertEndChild(pd);
  pElement->InsertEndChild(pose);


  XMLElement *score = xmlDoc->NewElement("score");
  score->SetText(0.999556);
  pElement->InsertEndChild(score);

  XMLElement *occ = xmlDoc->NewElement("occ");
  occ->SetText(1);
  pElement->InsertEndChild(occ);

  pRoot->InsertEndChild(pElement);

  XMLElement *pElement1 = xmlDoc->NewElement("element");
  XMLElement *category1 = xmlDoc->NewElement("category");
  category1->SetText("lane");
  pElement1->InsertEndChild(category1);
  pRoot->InsertEndChild(pElement1);

  XMLElement *pElement2 = xmlDoc->NewElement("element");
  XMLElement *category2 = xmlDoc->NewElement("category");
  category2->SetText("arrow");
  pElement2->InsertEndChild(category2);
  pRoot->InsertEndChild(pElement2);

  XMLElement *pElement3 = xmlDoc->NewElement("element");
  XMLElement *category3 = xmlDoc->NewElement("category");
  category3->SetText("stop_line");
  pElement3->InsertEndChild(category3);
  pRoot->InsertEndChild(pElement3);

  xmlDoc->SaveFile("gpal_semantic_map.xml");

  return 0;
}